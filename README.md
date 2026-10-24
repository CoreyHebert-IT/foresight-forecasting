# Foresight

Hourly AI price forecasts for a web dashboard's forecast panel, built on GitHub Actions and a Cloudflare Worker.

Every hour, GitHub Actions runs the open-source [Kronos-small](https://github.com/shiyu-coder/Kronos) model (MIT license) on CPU for each ticker in parallel. Kronos was trained on candles from 45 exchanges. Each job feeds the model the last 512 one-hour candles and samples 20 possible paths for the next 24 candles. A final job merges the results and POSTs them to your feeds Worker (`FORECAST_URL`, for example `https://<your-worker>/forecast`). The Worker checks the document and stores it in KV, and the page reads it from `GET /forecast`.

This is model output, a statistical guess from past prices. It is not investment advice.

## Where the candles come from

| Tickers | First choice | If that fails |
| --- | --- | --- |
| Crypto pairs (BTC, ETH-USD, ...) | Coinbase Exchange's public candle API: official, free, no key | Yahoo Finance |
| Stocks and ETFs | Yahoo Finance's chart API: unofficial, no key | Schwab, through your feeds Worker |

The Schwab fallback works without the job ever holding your Schwab login. The job asks the feeds Worker's `/forecast/candles` route using the read-only `FORECAST_READ_TOKEN`. The feeds Worker then asks a separate Worker that holds the Schwab login, over a private Cloudflare service binding. Schwab returns 30-minute bars, and the feeds Worker combines them into hours on the same 9:30, 10:30, ... grid Yahoo uses.

If the weekly Schwab sign-in lapses, the fallback just fails and Yahoo keeps working. The panel's status chip tooltip shows which source each ticker used.

That Worker exposes a market-data-only RPC entrypoint whose one method returns intraday candles. The feeds Worker binds to that entrypoint and nothing else, so it can't reach positions or orders.

Deploy the Schwab Worker before the feeds Worker so the binding works as soon as the feeds Worker is live. If it's ever missing or out of date, the fallback just reports an error and Yahoo keeps working.

## Setup (once)

1. Create a new public GitHub repository and upload everything in this folder. Keep the `.github` folder, since that's where the schedule lives. It has to be public because this hourly schedule would use up a private repository's free Actions minutes in under a week, while public repositories get unlimited minutes. Nothing secret is stored in the code.
2. Make two different long random tokens, for example with `openssl rand -hex 32` twice, or any password generator set to 64 characters:
   - `FORECAST_TOKEN` publishes forecasts. Only the final publish job gets it.
   - `FORECAST_READ_TOKEN` can only read the on-demand ticker list and Schwab fallback candles. The per-ticker jobs get this one, so even a compromised package in those jobs couldn't publish fake forecasts.
3. Give each token to both sides:
   - Worker: from the feeds Worker's folder, run `npx wrangler secret put FORECAST_TOKEN` and `npx wrangler secret put FORECAST_READ_TOKEN`, pasting the matching token each time.
   - GitHub: open the repo's Settings > Secrets and variables > Actions. Under Secrets, add both with the same values. Under Variables, add `FORECAST_URL` = your Worker's forecast URL, for example `https://<your-worker>/forecast`.
4. Open the Actions tab, select Foresight forecasts, then click Run workflow to run it once by hand. After that it runs at minute 7 of every hour.
5. Check your Worker's `/health` route. The `forecast` field should show the run time and the number of tickers.

## On-demand forecasts for any ticker

Ask the forecast panel for a ticker that isn't on the list, for example AMD, and it offers a RUN FORECAST FOR AMD button. It opens the Worker's `/forecast/request?sym=AMD` page in a new tab. Cloudflare Access asks you to sign in, you confirm, and the Worker starts this workflow for that one ticker. The forecast shows up in the panel about 3 to 5 minutes later, and the ticker stays in the hourly run for 24 hours. A ticker Yahoo doesn't know shows "could not be forecast" with the reason.

To set it up once:

1. Create a Cloudflare Access app. In Cloudflare Zero Trust, go to Access > Applications > Add an application > Self-hosted.
   - Domain: your Worker's hostname, path: `forecast/request`.
   - Policy: Allow, and include Emails = your email.
   - Save it, then copy the app's Application Audience (AUD) Tag, and note your team domain (`<team>.cloudflareaccess.com`, under Settings > Custom Pages or in the login URL).
2. Make a GitHub token. Go to GitHub Settings > Developer settings > Fine-grained tokens > Generate new token.
   - Repository access: only this forecast repo.
   - Permissions: Actions, Read and write. Nothing else.
3. Add the Worker settings. From the feeds Worker's folder, run each line and paste the value it asks for:

   ```
   npx wrangler secret put GITHUB_TOKEN          # the fine-grained token
   npx wrangler secret put FORECAST_REPO         # owner/repo, e.g. your-name/foresight-forecasting
   npx wrangler secret put ACCESS_TEAM_DOMAIN    # e.g. yourteam.cloudflareaccess.com
   npx wrangler secret put ACCESS_AUD            # the AUD tag from step 1
   npx wrangler secret put FORECAST_ADMIN_EMAIL  # required: your email; only this login can start runs
   ```

4. Check your Worker's `/health` route. Under `forecast`, it should say `"onDemand": "configured"`.

The Worker checks the Access sign-in token itself and requires it to be your email (`FORECAST_ADMIN_EMAIL`). If the Access app is ever missing, misconfigured or opened up to more people, the page still refuses everyone but you. The confirm button is a same-site POST, so a link on another site can't start a run. There's a cap of 25 on-demand runs a day, and at most 10 extra tickers are kept in the hourly run.

## Changing tickers

Edit `DEFAULT_TICKERS` in `forecast/forecast.py`. The workflow's plan step reads that list, so nothing else needs to change. Each ticker is its own parallel job, so adding one doesn't make a run take longer. Keep the defaults plus on-demand extras at 20 or fewer, which is GitHub's free limit on jobs running at the same time.

## Knobs

These are set as environment variables in the workflow, or locally when testing:

| Variable | Default | What it does |
| --- | --- | --- |
| `SAMPLES` | 20 | Sampled future paths per ticker. Runtime grows in step with it. |
| `LOOKBACK` | 512 | Past candles fed to the model (512 is Kronos-small's maximum). |
| `HORIZON` | 24 | Candles forecast ahead. |

## Run it locally

Use Python 3.12 on 64-bit Linux, the platform the lock file was built for. On other systems, install from `requirements.in` instead.

```
pip install --require-hashes --no-deps --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu -r forecast/requirements.lock
python forecast/forecast.py run --tickers NVDA --out nvda.json
```

On a 2-core machine, one ticker with 20 samples takes about 4 minutes. A GitHub runner has 4 cores, so it's roughly twice as fast.

## Notes

- Stock forecast times follow the hours the market actually traded over the last month, so overnight and weekend gaps are skipped automatically. That also adapts on its own when 23/5 trading starts. Holidays aren't skipped.
- See "Supply chain" below for how each outside piece is pinned.
- GitHub sometimes starts scheduled runs late when it's busy.

## Supply chain

Nothing this job runs is fetched as "latest", so a compromised upstream can't slip new code into it:

| Piece | How it's protected |
| --- | --- |
| Kronos model code | A reviewed copy lives in `vendor/kronos` (MIT license). It's copied from upstream commit `67b630e`, with per-file hashes in `vendor/kronos/PROVENANCE.md`. Nothing is cloned at run time. |
| Model weights | A copy can be kept in your own Cloudflare R2 bucket, served by the feeds Worker at `/weights/...` on the same host as `FORECAST_URL` (or set `WEIGHTS_MIRROR`). Upload it once with a script that checks each file's hash before uploading. The job downloads from your mirror first and falls back to Hugging Face at the same pinned commit. Every file is checked against the pinned SHA-256 hashes (`PINNED_WEIGHTS` in `forecast.py`) before loading, so a mismatch from either source stops the run. Only `safetensors` files are used, which can't execute code. |
| Python packages | They install from `requirements.lock` with `--require-hashes`, so every package, including indirect ones, must match its recorded hash. The lock only includes releases at least two weeks old when it was built. |
| GitHub Actions | Every action is pinned to a full commit SHA, with the version tag noted in a comment. |
| Secrets | `FORECAST_TOKEN` exists only in the repo's Actions secrets. The workflow runs with read-only repo permissions, and ticker input reaches scripts through environment variables, never pasted into shell commands. |

To update any of these, change it on purpose: review the upstream diff, then update the copy, hash or SHA in one commit.
