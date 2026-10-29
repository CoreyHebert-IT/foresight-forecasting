"""
Foresight: AI price forecasts for a dashboard's forecast panel, built on the
open-source Kronos-small model (MIT, see vendor/kronos).

Runs every hour on GitHub Actions (see .github/workflows/forecast.yml):
  run    one job per ticker, in parallel: pulls recent 1-hour candles from Yahoo
         Finance, feeds the last LOOKBACK candles to Kronos-small and samples
         SAMPLES possible paths for the next HORIZON candles. Writes part.json.
  merge  collects the parts and sends one document (quantile bands, a few sample
         paths, chance of closing higher) to the feeds Worker at FORECAST_URL,
         which stores it in KV and serves it to the page at GET /forecast.

  python forecast.py plan [--tickers AMD]          (prints the ticker list as JSON)
  python forecast.py run --tickers NVDA --out part.json
  python forecast.py merge parts/ --publish

Tickers outside DEFAULT_TICKERS work too (any Yahoo-listed US stock or ETF, or a
crypto pair like ETH-USD). The Worker keeps tickers requested from the forecast panel
for 24 hours, and `plan` adds them to the hourly run.
  python forecast.py run --out forecast.json        (every ticker, one machine)

Environment:
  FORECAST_URL    e.g. https://<your-worker>/forecast  (required to publish)
  FORECAST_TOKEN  publish secret, same value as the Worker secret (merge --publish only)
  FORECAST_READ_TOKEN  read-only secret for /forecast/extras and /forecast/candles
  KRONOS_DIR      Kronos model code (default: the reviewed copy in vendor/kronos)
  WEIGHTS_MIRROR  weights mirror (default: /weights on the FORECAST_URL host)
  SAMPLES, LOOKBACK, HORIZON, TICKERS  optional overrides

Model output only. It is a statistical guess from past candles, not advice.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

# numpy and pandas are imported in cmd_run, so `plan` and `merge` run on plain
# Python with nothing installed.
np = pd = None

HERE = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.environ.get("KRONOS_DIR", os.path.join(HERE, "..", "vendor", "kronos"))

# display symbol, Yahoo symbol, TradingView symbol (for the GP link), name
DEFAULT_TICKERS = [
    ("NVDA",  "NVDA",  "NASDAQ:NVDA",  "Nvidia"),
    ("AMD",   "AMD",   "NASDAQ:AMD",   "AMD"),
    ("MSFT",  "MSFT",  "NASDAQ:MSFT",  "Microsoft"),
    ("AMZN",  "AMZN",  "NASDAQ:AMZN",  "Amazon"),
    ("GOOGL", "GOOGL", "NASDAQ:GOOGL", "Alphabet"),
    ("TSLA",  "TSLA",  "NASDAQ:TSLA",  "Tesla"),
    ("SPCX",  "SPCX",  "NASDAQ:SPCX",  "SpaceX"),
    ("CVX",   "CVX",   "NYSE:CVX",     "Chevron"),
    ("XOM",   "XOM",   "NYSE:XOM",     "ExxonMobil"),
    ("PLTR",  "PLTR",  "NASDAQ:PLTR",  "Palantir"),
    ("VXUS",  "VXUS",  "NASDAQ:VXUS",  "Total International ETF"),
    ("VOO",   "VOO",   "AMEX:VOO",     "S&P 500 ETF"),
]

MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"

# Supply-chain pinning: the weights are fetched from Hugging Face at an exact
# commit and every file is checked against these SHA-256 hashes before it is
# loaded. If the upstream repo is changed, deleted or tampered with, the run
# stops instead of loading something different. Only safetensors weights are
# used, which hold plain numbers and cannot run code (no pickle).
PINNED_WEIGHTS = {
    TOKENIZER_ID: ("0e0117387f39004a9016484a186a908917e22426", {
        "config.json": "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
        "model.safetensors": "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
    }),
    MODEL_ID: ("901c26c1332695a2a8f243eb2f37243a37bea320", {
        "config.json": "5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
        "model.safetensors": "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
    }),
}


# Weights come from your own mirror first (R2 behind the feeds Worker, served at
# /weights on the same host as FORECAST_URL unless WEIGHTS_MIRROR says otherwise),
# then Hugging Face as a fallback. Either way the bytes must match PINNED_WEIGHTS,
# so the source only matters for availability.
def _default_mirror():
    u = urllib.parse.urlsplit(os.environ.get("FORECAST_URL", ""))
    return f"https://{u.netloc}/weights" if u.scheme == "https" and u.netloc else ""


WEIGHTS_MIRROR = os.environ.get("WEIGHTS_MIRROR", _default_mirror()).rstrip("/")
WEIGHTS_DIR = os.path.expanduser(os.environ.get("WEIGHTS_DIR", "~/.cache/foresight-weights"))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def verified_weights(repo_id):
    """Local folder holding the pinned, hash-checked files for repo_id."""
    revision, files = PINNED_WEIGHTS[repo_id]
    folder = os.path.join(WEIGHTS_DIR, repo_id.replace("/", "--"), revision)
    os.makedirs(folder, exist_ok=True)
    for name, want in files.items():
        dest = os.path.join(folder, name)
        if os.path.exists(dest) and _sha256(dest) == want:
            continue
        sources = []
        if WEIGHTS_MIRROR.startswith("https://"):
            sources.append(("mirror", f"{WEIGHTS_MIRROR}/{repo_id}/{revision}/{name}"))
        sources.append(("huggingface", f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}"))
        for label, url in sources:
            try:
                _download(url, dest)
            except Exception as e:
                print(f"{repo_id}/{name}: {label} unavailable ({e})", file=sys.stderr)
                continue
            got = _sha256(dest)
            if got == want:
                print(f"{repo_id}/{name}: verified from {label}", flush=True)
                break
            print(f"{repo_id}/{name}: {label} copy has the WRONG hash {got}, discarding", file=sys.stderr)
            os.remove(dest)
        else:
            sys.exit(f"{repo_id}/{name}: no source had the pinned file, refusing to run")
    return folder


LOOKBACK = int(os.environ.get("LOOKBACK", 512))   # Kronos-small's max context
HORIZON = int(os.environ.get("HORIZON", 24))      # candles ahead
SAMPLES = int(os.environ.get("SAMPLES", 20))      # sampled future paths per ticker
MIN_HISTORY = 200   # Schwab keeps less intraday history than Yahoo
HIST_OUT = 96                                      # past candles sent to the page (the chart shows at most 96)
PATHS_OUT = 12                                     # sample paths sent to the page
UA = "Mozilla/5.0 (compatible; ForesightForecast/1.0)"


SYM_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z0-9]{1,3})?$|^[A-Z0-9]{2,10}-USD$")


def tickers(arg=""):
    """(display, yahoo, tradingview, name) for each requested symbol. Known tickers
    use DEFAULT_TICKERS; any other valid symbol is looked up on Yahoo as-is
    (BRK.B becomes BRK-B there)."""
    only = [t.strip().upper() for t in (arg or os.environ.get("TICKERS", "")).split(",") if t.strip()]
    if not only:
        return list(DEFAULT_TICKERS)
    known = {t[0]: t for t in DEFAULT_TICKERS}
    out = []
    for sym in dict.fromkeys(only):
        if sym in known:
            out.append(known[sym])
        elif SYM_RE.match(sym):
            ysym = sym if sym.endswith("-USD") else sym.replace(".", "-")
            out.append((sym, ysym, None, sym))
        else:
            print(f"skipping {sym!r}: not a ticker", file=sys.stderr)
    return out


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib re-sends custom headers, Authorization included, when it follows a redirect, even to
    another host. Requests that carry a token don't follow redirects at all."""
    def redirect_request(self, *args, **kwargs):
        return None


def _authed_open(req, timeout):
    if not req.full_url.lower().startswith("https://"):
        raise RuntimeError("refusing to send a token to a non-https URL")
    return urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout)


def _get_json(url, headers=None, limit=4_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
    authed = any(k.lower() == "authorization" for k in (headers or {}))
    with (_authed_open(req, 20) if authed else urllib.request.urlopen(req, timeout=20)) as r:
        return json.loads(r.read(limit))


def _frame(ts_sec, o, h, l, c, v, tz):
    df = pd.DataFrame({
        "ts": pd.to_datetime(ts_sec, unit="s", utc=True).tz_convert(tz),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    }).dropna()
    df["volume"] = df["volume"].astype(float)
    df = df.drop_duplicates("ts").sort_values("ts")
    # After the close Yahoo appends a short 16:00 stub to the 9:30, 10:30 ... grid.
    # Keep only candles on the usual minute so every row is a full hour.
    if len(df):
        df = df[df["ts"].dt.minute == df["ts"].dt.minute.mode().iloc[0]]
    # Drop the candle that is still forming.
    now = pd.Timestamp.now(tz=tz)
    df = df[df["ts"] + pd.Timedelta(minutes=60) <= now + pd.Timedelta(minutes=1)]
    return df.reset_index(drop=True)


def yahoo_candles(ysym):
    """1-hour candles from Yahoo's chart API (unofficial, no key), exchange local time."""
    data = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ysym)}"
                     "?interval=60m&range=6mo&includePrePost=false")
    res = data["chart"]["result"][0]
    meta, q = res["meta"], res["indicators"]["quote"][0]
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    name = (meta.get("shortName") or meta.get("longName") or "").strip()[:60]
    return _frame(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"], tz), tz, name


def coinbase_candles(product, want=LOOKBACK + 8):
    """1-hour candles from Coinbase Exchange's public API (official, no key).
    300 candles per request, newest first: [time, low, high, open, close, volume]."""
    rows, end = [], int(time.time())
    while len(rows) < want:
        start = end - 300 * 3600
        iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
        part = _get_json(f"https://api.exchange.coinbase.com/products/{urllib.parse.quote(product)}/candles"
                         f"?granularity=3600&start={iso(start)}&end={iso(end)}")
        if not isinstance(part, list) or not part:
            break
        rows += part
        end = start
    if not rows:
        raise RuntimeError("Coinbase returned no candles")
    t, lo, hi, op, cl, vo = zip(*[r[:6] for r in rows])
    return _frame(list(t), op, hi, lo, cl, vo, "UTC"), "UTC", ""


def schwab_candles(sym):
    """Fallback: Schwab 30-minute candles combined into hours by the feeds Worker,
    which reaches the Schwab Worker over a private service binding. The Schwab login
    tokens never leave Cloudflare; this job only holds the read-only
    FORECAST_READ_TOKEN (the publish token is given to the publish job alone)."""
    url, token = os.environ.get("FORECAST_URL"), os.environ.get("FORECAST_READ_TOKEN")
    if not (url and token):
        raise RuntimeError("Schwab fallback needs FORECAST_URL and FORECAST_READ_TOKEN")
    d = _get_json(f"{url.rstrip('/')}/candles?sym={urllib.parse.quote(sym)}", {"Authorization": "Bearer " + token})
    c = d.get("candles") or []
    if not c:
        raise RuntimeError(d.get("error") or "Schwab returned no candles")
    t, o, h, l, cl, v = zip(*c)
    tz = d.get("tz") or "America/New_York"
    return _frame(list(t), o, h, l, cl, v, tz), tz, ""


def fetch_candles(display, ysym):
    """Crypto: Coinbase, then Yahoo. Stocks and ETFs: Yahoo, then Schwab.
    Returns (frame, tz, name, source)."""
    is_crypto = ysym.endswith("-USD")
    chain = [("coinbase", lambda: coinbase_candles(ysym)), ("yahoo", lambda: yahoo_candles(ysym))] if is_crypto \
        else [("yahoo", lambda: yahoo_candles(ysym)), ("schwab", lambda: schwab_candles(display))]
    problems = []
    for label, get in chain:
        try:
            df, tz, name = get()
        except Exception as e:
            problems.append(f"{label}: {str(e)[:80]}")
            continue
        if len(df) >= MIN_HISTORY:
            if problems:
                print(f"{display}: using {label} ({'; '.join(problems)})", flush=True)
            return df, tz, name, label
        problems.append(f"{label}: only {len(df)} candles")
    if all("404" in p for p in problems[:1]) and not is_crypto:
        raise RuntimeError("ticker not found on Yahoo Finance" + ("" if len(problems) < 2 else f"; {problems[1]}"))
    raise RuntimeError("; ".join(problems))


def future_stamps(ts, n):
    """Next n candle start times, using only weekday/hour slots the market actually
    traded in over the last month. This follows the real session automatically,
    including the move to 23/5 trading, without a hard-coded calendar."""
    recent = ts[ts >= ts.iloc[-1] - pd.Timedelta(days=31)]
    slots = {(t.weekday(), t.hour, t.minute) for t in recent}
    out, t = [], ts.iloc[-1]
    for _ in range(24 * 14):
        t = t + pd.Timedelta(hours=1)
        if (t.weekday(), t.hour, t.minute) in slots:
            out.append(t)
            if len(out) == n:
                break
    return pd.Series(out)


def rnd(x, ref):
    d = 2 if ref >= 10 else 4 if ref >= 0.1 else 6
    return round(float(x), d)


def run_one(predictor, display, ysym, tvsym, name):
    df, tz, yname, source = fetch_candles(display, ysym)
    if name == display and yname:
        name = yname
    df = df.iloc[-LOOKBACK:].reset_index(drop=True)
    x_df = df[["open", "high", "low", "close", "volume"]]
    x_ts = df["ts"].dt.tz_localize(None)
    y_ts = future_stamps(df["ts"], HORIZON)
    if len(y_ts) < HORIZON:
        raise RuntimeError("could not build future timestamps")
    y_ts_naive = y_ts.dt.tz_localize(None)

    # SAMPLES copies of the same series in one batch = SAMPLES independent paths.
    preds = predictor.predict_batch(
        df_list=[x_df] * SAMPLES, x_timestamp_list=[x_ts] * SAMPLES, y_timestamp_list=[y_ts_naive] * SAMPLES,
        pred_len=HORIZON, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    closes = np.stack([p["close"].values for p in preds])           # (SAMPLES, HORIZON)
    last = float(df["close"].iloc[-1])
    qs = {f"p{q}": [rnd(v, last) for v in np.percentile(closes, q, axis=0)] for q in (10, 25, 50, 75, 90)}
    final = closes[:, -1]
    hist = df.iloc[-HIST_OUT:]
    return {
        "sym": display, "tv": tvsym, "name": name, "tz": tz, "interval": "1h",
        "last": {"t": int(df["ts"].iloc[-1].timestamp()), "c": rnd(last, last)},
        "hist": [[int(r.ts.timestamp()), rnd(r.open, last), rnd(r.high, last), rnd(r.low, last), rnd(r.close, last)]
                 for r in hist.itertuples()],
        "steps": [int(t.timestamp()) for t in y_ts],
        "q": qs,
        "paths": [[rnd(v, last) for v in closes[i]] for i in range(min(PATHS_OUT, SAMPLES))],
        "pUp": round(float((final > last).mean()), 3),
        "medMovePct": round(float((np.median(final) / last - 1) * 100), 2),
        "samples": SAMPLES, "src": source,
    }


def cmd_run(args):
    global np, pd
    import numpy as np
    import pandas as pd
    sys.path.insert(0, KRONOS_DIR)
    from model import Kronos, KronosTokenizer, KronosPredictor  # from the vendored copy (KRONOS_DIR)
    t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained(verified_weights(TOKENIZER_ID))
    model = Kronos.from_pretrained(verified_weights(MODEL_ID))
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    items, errors = [], []
    for display, ysym, tvsym, name in tickers(args.tickers):
        t1 = time.time()
        try:
            items.append(run_one(predictor, display, ysym, tvsym, name))
            print(f"{display}: ok in {time.time() - t1:.1f}s", flush=True)
        except Exception as e:  # one bad ticker must not sink the run
            errors.append({"sym": display, "err": str(e)[:150]})
            print(f"{display}: FAILED {e}", file=sys.stderr, flush=True)
    doc = build_doc(items, errors, time.time() - t0)
    write(args.out, doc)
    if args.publish:
        publish(doc)
    # A failed ticker is reported as an error (so the panel can say why) rather
    # than failing the job; only a run with nothing at all is an error.
    if not items and not errors:
        sys.exit("no tickers to forecast")


def cmd_plan(args):
    """Ticker list for the GitHub Actions matrix, as JSON on stdout. A manual or
    on-demand run passes --tickers; the hourly run uses the defaults plus anything
    requested from the forecast panel in the last 24 hours."""
    if args.tickers:
        syms = [t[0] for t in tickers(args.tickers)]
    else:
        syms = [t[0] for t in DEFAULT_TICKERS]
        url, token = os.environ.get("FORECAST_URL"), os.environ.get("FORECAST_READ_TOKEN")
        if url and token:
            try:
                req = urllib.request.Request(url.rstrip("/") + "/extras", headers={
                    "Authorization": "Bearer " + token, "User-Agent": UA, "Accept": "application/json"})
                with _authed_open(req, 15) as r:
                    extra = json.loads(r.read(20000)).get("syms", [])
                syms += [t[0] for t in tickers(",".join(extra)) if t[0] not in syms][:10]
            except Exception as e:  # the hourly run still goes ahead with the defaults
                print(f"could not read on-demand tickers: {e}", file=sys.stderr)
    print(json.dumps(syms[:20]))


def cmd_merge(args):
    items, errors, secs = [], [], 0.0
    for path in sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                part = json.load(f)
            items += part.get("items", [])
            errors += part.get("errors", [])
            secs = max(secs, part.get("runSeconds", 0))
        except Exception as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
    order = [t[0] for t in DEFAULT_TICKERS]
    items.sort(key=lambda i: order.index(i["sym"]) if i.get("sym") in order else 99)
    doc = build_doc(items, errors, secs)
    write(args.out, doc)
    if not items and not errors:
        sys.exit("nothing to publish")
    if args.publish:
        publish(doc)


def build_doc(items, errors, secs):
    return {
        "v": 1, "generated": int(time.time() * 1000), "model": MODEL_ID, "lookback": LOOKBACK,
        "horizon": HORIZON, "interval": "1h", "items": items, "errors": errors, "runSeconds": round(secs, 1),
    }


def write(path, doc):
    body = json.dumps(doc, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"{path}: {len(doc['items'])} forecasts, {len(doc['errors'])} errors, {len(body) / 1024:.0f} KB", flush=True)


def publish(doc):
    url, token = os.environ.get("FORECAST_URL"), os.environ.get("FORECAST_TOKEN")
    if not (url and token):
        sys.exit("FORECAST_URL and FORECAST_TOKEN must both be set to publish")
    if not url.startswith("https://"):
        sys.exit("FORECAST_URL must be https")
    req = urllib.request.Request(url, data=json.dumps(doc, separators=(",", ":")).encode(), method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + token, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        print("published:", r.status, r.read(200).decode(errors="replace"), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="forecast tickers")
    r.add_argument("--tickers", default="", help="comma list of display symbols, default all")
    r.add_argument("--out", default="forecast.json")
    r.add_argument("--publish", action="store_true")
    pl = sub.add_parser("plan", help="print the ticker list for the workflow matrix")
    pl.add_argument("--tickers", default="")
    m = sub.add_parser("merge", help="combine part files and publish")
    m.add_argument("dir")
    m.add_argument("--out", default="forecast.json")
    m.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    {"run": cmd_run, "merge": cmd_merge, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    main()
