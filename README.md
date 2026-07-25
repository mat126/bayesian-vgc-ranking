# Bayesian Ranking for Pokémon VGC

A hierarchical Bayesian Bradley-Terry model that estimates which Pokémon and which teams enter
the 2026 World Championships (Pokémon Champions, Regulation Set M-B) as favourites, plus a
low-rank synergy variant and a neural Set Transformer for comparison.

Team strength is not a free parameter, which the combinatorics make impossible, but
**decomposed into the effects of its components**: species, moves, abilities, items and
natures, plus a player skill effect. Inference is hierarchical Bayesian, with partial pooling
regularising rare levels and credible intervals on every estimate. All headline numbers below
come from NUTS (Hamiltonian Monte Carlo); read them **by interval, not by mean**.

## Key results

Dataset: **17,581 matches** from Limitless tournaments in Regulation M-B, complete teamlists on
both sides, 3,878 distinct players, 5,393 unique teams (only 95 seen five or more times — which
is exactly why feature decomposition is necessary rather than a refinement).

### The pilot dominates the team sheet

Posterior block scales (`tau`), additive model:

| Block | tau (mean) | interpretation |
|---|---|---|
| **player** | **0.550** [0.50, 0.60] | dominant and precisely estimated |
| ability | 0.107 | |
| item | 0.102 | team-feature blocks, statistically |
| species | 0.096 | indistinguishable from one another |
| moves | 0.077 | |
| nature | 0.058 | smallest |

Player skill carries roughly **five times** the dispersion of any team feature. Who plays
predicts outcomes far more than what they play. The caveat: pilot skill and team strength
separate only through the player × team crossing in the data, so the exact ratio is not a clean
causal decomposition, though the size of the gap makes the qualitative conclusion robust. At the
individual level, **141 of 1,657 modelled players** have a skill effect whose interval clears
zero — while **zero of 219 species** do (see below).

### No single Pokémon is a robust best pick

Under the full posterior, **not one of 219 species** has an effect whose 90% interval clears
zero. The leaders (Kangaskhan 0.107, Archaludon 0.102, Eternal Flower Floette 0.100, Charizard
0.096) are plausible but not established. This overturns what ADVI had suggested: variational
inference underestimated the uncertainty, and the honest NUTS posterior says the marginal effect
of one species slot out of six is simply small next to the pilot and the noise floor. No species
is an auto-win button; value lives in whole configurations.

### Where the signal is: moves, items, natures

Identifiability improves from "which Pokémon" to "which choices". **Twelve moves** clear zero,
led by Encore (0.121), Weather Ball (0.115), Quick Guard (0.112) and Swords Dance (0.107) —
control and utility over raw power. **Seven items** clear zero, led by Venusaurite (0.143), with
Mega Stones dominating the block as expected in a Mega-Evolution format without Terastallization.
Three natures are robust (Calm, Modest, Bold), all defensive or special-leaning. Two abilities
(Unnerve, Swift Swim).

### Strongest teams

Team strength `s(T)` (pilot excluded) for the 95 archetypes seen at least five times is led by a
**Sneasler / Kingambit / Incineroar / Sinistcha core** in several fifth/sixth-slot variations.
The synergy model reproduces the same podium. The head-to-head between the top two teams is
**54.7% [47.6%, 62.2%]** at equal pilot skill — a real edge that brushes a coin flip. No cliffs.

### Synergy is real, small, and concentrated

Of 1,936 co-occurring species pairs, exactly **four** have robustly positive synergy:
Archaludon + Grimmsnarl (0.095), Kingambit + Sneasler (0.091), Kingambit + Sinistcha (0.089),
Eternal Flower Floette + Kingambit (0.080). Kingambit anchors three of the four — the model
learned the format's dominant core from win/loss data alone.

### Did non-linearity help? No — and that is a result

Held-out log-loss, the metric that decides which model predicts best:

| Model | held-out log-loss | held-out accuracy |
|---|---|---|
| Coin flip | 0.6931 | 0.500 |
| Bayesian additive (ADVI) | 0.6618 | 0.614 |
| **Bayesian additive (NUTS)** | **0.6584** | 0.601 |
| Neural Set Transformer | 0.6692 | ~0.61 |

The rigorous NUTS refit gives the best held-out log-loss (0.6584), and the neural network does
**not** beat it. The Set Transformer was the right architecture to let higher-order synergy
express itself, trained with the right regularisation for the data scale, and it found nothing
the additive model plus degree-2 synergy had missed. At 17k matches, whatever higher-order
structure exists in the metagame is not learnable past the noise floor set by unobserved
in-battle decisions and the pilot effect. The regularised Bayesian model is not a baseline the
deep model failed to beat; it is the right-sized model for the problem.

Even the best model sits only modestly below chance: a single VGC match stays close to a coin
flip even given both full teamlists and pilot history. The unmodelled residue is the game itself
— team preview reading, in-battle decisions, and RNG.

Figures in [`figures/`](figures/), tables in [`results/`](results/), full write-up in the
[blog post](vgc_ranking_blog.qmd).

## Repository layout

```
bayesian-ranking-vgc/
├── README.md
├── environment.yml           reproducible conda environment
├── .gitignore
├── vgc_ranking_blog.qmd       long-form technical write-up
├── src/                       all code, flat so modules import each other
├── notebooks/                 analysis, comparison, and Kaggle GPU notebooks
├── docs/                      architecture diagram
├── figures/                   versioned plots
├── results/                   versioned ranking tables
├── data/                      [not versioned] regenerable from the pipeline
└── models/                    [not versioned] posteriors and checkpoints
```

Each directory carries its own README. Scripts live in a flat `src/` so cross-module imports
work without packaging: running `python src/name.py` from the root puts `src/` at the head of
the path while relative data paths resolve from the root.

## Pipeline

```bash
conda env create -f environment.yml
conda activate vgc

# 1. collect (public Limitless API, no key)
python src/limitless_vgc.py --format <FORMAT_ID_REG_M_B> --min-players 30 --max-tournaments 300

# 2. clean (multilingual canonicalisation)
python src/canonicalize.py data/matches.jsonl data/matches_clean.jsonl --fuzz 85

# 3. fit — additive and synergy, definitive NUTS estimates
python src/bt_bayes.py --matches data/matches_clean.jsonl --method nuts --sampler nutpie \
    --draws 800 --tune 500 --chains 2 --min-freq 5 --min-player-matches 8 --target-accept 0.8
python src/bt_bayes.py --matches data/matches_clean.jsonl --method nuts --synergy-dim 4 --tag nuts_syn4

# 4. neural alternative
python src/neural_train.py --matches data/matches_clean.jsonl --epochs 80
python src/neural_eval.py --ckpt models/neural_ckpt.pt --matches data/matches_clean.jsonl --teams data/teams.jsonl
```

GPU sampling (an order of magnitude faster on this workload) runs through the Kaggle notebooks
in [`notebooks/`](notebooks/), also published on
[kaggle.com/mat126](https://www.kaggle.com/mat126). See [`src/README.md`](src/README.md) for the
full flag reference and the performance-tuning history.

## Modelling notes

**Likelihood.** Bradley-Terry Bernoulli: `logit P(A beats B) = s(A) − s(B)`, difference
structure enforcing antisymmetry. A side intercept absorbs the player1 advantage (51.3%).

**Structured strength.** `s(T)` sums six Pokémon contributions, each a sum of feature effects.
Being linear in counts, the strength difference is the coefficients times the difference of the
teams' count vectors — hence the design matrix.

**Identifiability and geometry.** Six species per team make an additive per-block constant
cancel; `ZeroSumNormal` removes it. That constraint is non-centred only with respect to itself,
**not** with respect to the hierarchical scale, so the model multiplies a standardised variable
by `tau` explicitly — without this, the funnel froze chains at step size 0.001.

**Conditioning.** Players below eight matches are pooled to a shared zero effect (their matches
stay in the likelihood; only the individual skill parameter is dropped). This kept 1,657 players
and turned a sampler saturating its tree depth into one that runs.

**Synergy.** Optional low-rank factorisation machine: one latent vector per species, pairwise
synergy as an inner product in closed form, with no self-synergy by construction.

**Neural model.** Siamese Set Transformer: shared tower scoring each team, difference giving the
logit (antisymmetry by construction); double permutation invariance (moves, then Pokémon) via
mean-pooling and self-attention with no positional encoding. 106,738 parameters, deliberately
small.

## Known limitations

Stochastic transitivity (a 1-D strength scale cannot represent matchup triangles); player
confounding (the main interpretive caveat); non-stationarity (the metagame shifts weekly);
population (online Limitless data differs from the in-person Worlds field).

## Roadmap

- [x] Definitive NUTS estimation (GPU, numpyro)
- [x] Additive vs synergy comparison
- [x] Neural Set Transformer trained and compared
- [x] Rigorous NUTS held-out validation (0.6584)
- [ ] Fully aligned three-model comparison on one shared split (row-for-row)
- [ ] Temporal decay of match weights toward August
- [ ] Synergy extension to moves and items
- [ ] Continuous dataset updates through Worlds (28–30 August 2026)

## Links and data sources

- GPU notebooks on Kaggle: [kaggle.com/mat126](https://www.kaggle.com/mat126)
- [Limitless](https://play.limitlesstcg.com) — public API, primary match source
- [PokéAPI](https://github.com/PokeAPI/pokeapi) — multilingual name tables for canonicalisation
- [Pokémon Showdown](https://replay.pokemonshowdown.com) — optional secondary source

## Licence

Released under the MIT Licence. You are free to use, copy, modify and distribute this code,
including for commercial purposes, provided the copyright notice and licence text are retained.
The software is provided "as is", without warranty of any kind. See [`LICENSE`](LICENSE) for the
full text.

The MIT Licence covers the code in this repository. It does not extend to third-party data:
match data comes from the Limitless API and reference tables from PokéAPI, each under its own
terms, and Pokémon names and related trademarks belong to Nintendo, Game Freak and The Pokémon
Company. This is an independent, non-commercial research project with no affiliation to or
endorsement by those parties.

## Use of AI assistance

Parts of this project were developed with the help of an AI coding assistant, used for
scaffolding, refactoring, debugging, and drafting documentation. Every design decision — the
model structure, the choice of priors, the handling of identifiability and sampler geometry,
the interpretation of results — was made, reviewed and validated by me. AI-assisted code was
read, tested and corrected before being committed; it was a tool for moving faster, not a
substitute for understanding. Any errors that remain are my own.