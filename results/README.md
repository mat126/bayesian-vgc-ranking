# `results/` — versioned

Ranking tables are small and are the point of the project, so they are tracked. All current
files are from the **NUTS** posteriors.

| File | Content |
|---|---|
| `ranking_species_nuts.csv` | Species effects (the Pokemon ranking) |
| `ranking_item_nuts.csv` | Item effects (Mega Stones dominate) |
| `ranking_moves_nuts.csv` | Move effects |
| `ranking_ability_nuts.csv` | Ability effects |
| `ranking_nature_nuts.csv` | Nature effects |
| `ranking_teams_nuts.csv` | Team ranking (additive) |
| `ranking_teams_add.csv` / `ranking_teams_syn.csv` | Team ranking, additive vs synergy |
| `ranking_teams_neural.csv` | Team ranking from the neural model |
| `synergy_pairs.csv` | Species synergy pairs, strongest first |

## Columns

Feature files are indexed by level name with `mean`, `q5`, `q95` (5th/95th percentiles) and a
boolean `robust` (interval clears zero). Team files add `team` (the six species) and `n`
(occurrences). Synergy adds `species_1`, `species_2`, `n`.

## How to read these

**By interval, not by mean.** Zero of 219 species clear zero; a high mean with a wide interval
is noise, not a finding. Effects are in **log-odds**, identified up to an additive constant per
block, so read each relative to its block average. Team scores exclude the player component:
they answer "how strong is this team, holding the pilot constant", not "how often it wins".