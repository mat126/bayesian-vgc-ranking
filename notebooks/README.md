# `notebooks/`

## Local analysis

| Notebook | Purpose |
|---|---|
| `model_comparison.ipynb` | Additive vs synergy side by side: `tau`, effect rankings, team rankings for both, synergy pairs. |
| `neural_analysis.ipynb` | Neural model: held-out metrics, calibration, team ranking (MC-dropout), head-to-head, comparison table. Requires `neural_ckpt.pt`. |

Each notebook sets up paths in its first cell, so it runs whether opened from `notebooks/` or
the repository root. Outputs go to `figures/` and `results/`.

## Kaggle GPU

These run on GPU sessions and are published on [kaggle.com/mat126](https://www.kaggle.com/mat126).

| Notebook | Purpose |
|---|---|
| [`kaggle_nuts_gpu.ipynb`](https://www.kaggle.com/code/mat126/ranking-vgc) | Definitive NUTS sampling (additive and synergy) via the numpyro backend. |
| [`kaggle_neural_gpu.ipynb`](https://www.kaggle.com/code/mat126/vgc-neural) | Trains and evaluates the Set Transformer. |
| [`vgc_analysis.ipynb`](https://www.kaggle.com/code/mat126/vgc-analysis) |  Full Bayesian analysis from the saved posteriors: convergence diagnostics, `tau` decomposition, species / item / move / ability / nature rankings, player effects, team ranking, head-to-head, synergy, held-out validation.|

Two constraints that apply to every Kaggle notebook here:

**Never import `jax`, `numpyro` or `pymc` in the kernel.** JAX is multithreaded and shell
commands (`!`) use `os.fork()`; the combination deadlocks the kernel silently. Environment
checks and sampling run in subprocesses (`!python`), and only numpy/arviz are imported in-kernel.

**Sessions are capped at 12 hours** with no intermediate checkpointing. Run the short
calibration job first and extrapolate before launching a long one. Attach the private dataset
and enable GPU + Internet in Session options; the notebooks locate inputs by recursive glob, so
the exact mount path does not matter.