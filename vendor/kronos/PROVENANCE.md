# Vendored Kronos model code

Copied from https://github.com/shiyu-coder/Kronos at commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13), MIT license (see LICENSE).
Only the `model/` package needed for inference is included. Nothing was changed.

Reviewed 2026-10-24: the code only imports numpy, pandas, torch, einops, tqdm and
huggingface_hub, makes no network calls of its own, and uses no eval, exec,
subprocess or pickle.

SHA-256 of the copied files:
    f8f856ca3fedadcaac97e196be23d1aeda1c3c9ffe8903d66d43ea3bcac6240c  model/__init__.py
    0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032  model/kronos.py
    a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f  model/module.py
    acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc  LICENSE

To update: diff a newer upstream commit against these files, review the changes,
copy them in, and update the commit and hashes above. Model weights are pinned
separately (exact Hugging Face commit plus SHA-256) in forecast/forecast.py.
