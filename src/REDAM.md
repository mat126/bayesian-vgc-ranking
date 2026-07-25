# `src/` — source code

All modules live in a single flat directory on purpose. Running `python src/name.py` from the
repository root puts `src/` at the head of `sys.path`, so cross-module imports resolve without
packaging, while relative data paths resolve from the root.

## Module map

| Module | Role |
|---|---|
| `limitless_vgc.py` | Client for the public Limitless API. Joins `standings` (structured teamlists) with `pairings` (winners) into match records. |
| `pokepaste.py` | Parser for the pokepaste / Showdown export format. Fallback for non-API sources. |
| `inspect_standings.py` | Diagnostic: prints the real structure of a standings response. |
| `showdown_replays.py` | Optional secondary source (ladder, species level only). Different population — weight separately, never merge one-to-one with tournament data. |
| `canonicalize.py` | Maps dirty text fields to canonical English via slug, PokeAPI multilingual match, fuzzy match, and a manual override map. |
| `design_matrix.py` | Count-difference matrices per block, per-side species counts for synergy, player indices with optional threshold, outcome vector. |
| `bt_bayes.py` | The PyMC model, sampling, and posterior team-strength scoring. Imports `design_matrix`. |
| `neural_bt.py` | Siamese Set Transformer architecture. |
| `neural_data.py` | Tensor encoding, vocabularies (train split only), DataLoaders with A/B swap augmentation. |
| `neural_train.py` | Adam + weight decay + BCE, early stopping on validation log-loss. |
| `neural_eval.py` | Held-out metrics, calibration, team ranking with MC-dropout, head-to-head. |

## `bt_bayes.py` flag reference

```
--matches PATH              input JSONL
--method {advi,nuts,map}    inference method
--draws N                   posterior draws per chain
--min-freq N                frequency threshold for moves/ability/item/nature (species always kept)
--min-player-matches N      players below N matches pool to a shared zero effect (try 8);
                            greatly improves conditioning. No matches are discarded.
--limit N                   subsample N matches for fast tests
--synergy-dim D             latent dimension for species synergy (0 = off, 4 is reasonable)
--sparse                    sparse design matrices (~10x on CPU). NOT compatible with numpyro.
--sampler {pymc,nutpie,numpyro}
                            NUTS backend. nutpie = fast on CPU; numpyro = GPU, vectorises chains.
--tune N                    adaptation steps (500 is usually enough)
--chains N                  parallel chains (2 reduces contention on a laptop)
--target-accept F           0.9 conservative; 0.8 with zero divergences halves gradient evals/draw
--out-dir DIR / --tag NAME  outputs become bt_posterior_<tag>.nc and bt_artifacts_<tag>.json
```

## Performance history

Sampling this model is expensive and the bottleneck moved several times. The fixes, in case
they recur:

1. **Chains frozen at step size 0.001** — a hierarchical funnel. Fixed by explicit non-centred
   parameterisation (`ZeroSumNormal` is non-centred only w.r.t. its constraint, not `tau`).
2. **1023 gradient evaluations per draw** (tree depth 10 saturating) — nearly flat directions
   from rare players. Fixed by `--min-player-matches`.
3. **High per-gradient cost** with dense matrices — `--sparse` on CPU, or `--sampler numpyro`
   on GPU, which suits dense matrix-vector products.

Always run a short job first (`--draws 50 --limit 3000`) to measure seconds per draw. There is
no intermediate checkpointing: an interrupted sampling run is lost.