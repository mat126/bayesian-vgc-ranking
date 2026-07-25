# `models/` — not versioned

Posteriors and checkpoints are large (130-175 MB each) and fully regenerable, so they stay out
of git. Only this README is tracked. NetCDF files also exceed GitHub's size limits, so they must
not be committed.

## Naming convention

`bt_bayes.py` tags every run:

```
bt_posterior_<tag>.nc        posterior, read with arviz.from_netcdf
bt_artifacts_<tag>.json       vocabularies, player labels, and the run configuration
```

Tag defaults to `<method>_syn<synergy_dim>` (e.g. `nuts_syn0`, `nuts_syn4`), settable with
`--tag`. Key runs: `nuts_syn0` (additive), `nuts_syn4` (synergy), `heldout_train` (refit on the
85% split for validation).

## Why the artifacts file matters

It records the exact configuration each posterior was produced with: method, sampler, synergy
dimension, thresholds, tune, draws, chains, match count, players granted an effect. This exists
to prevent comparing models estimated under different settings as if equivalent. In particular,
`tau_player` under `--min-player-matches 8` is not comparable with one over all players — the
population differs.

## Neural checkpoints

```
neural_ckpt.pt               best weights by validation log-loss, plus vocabulary and config
neural_vocab.json            vocabulary alone, for standalone inference
```

## Regenerating

```bash
python src/bt_bayes.py --matches data/matches_clean.jsonl --method nuts --sampler nutpie \
    --draws 800 --tune 500 --chains 2 --min-freq 5 --min-player-matches 8 --target-accept 0.8
# GPU: notebooks/kaggle_nuts_gpu.ipynb
python src/neural_train.py --matches data/matches_clean.jsonl --epochs 80
```

No intermediate checkpointing during sampling — an interrupted run is lost. Calibrate runtime
with a short job first.