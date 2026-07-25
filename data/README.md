# `data/` — not versioned

Regenerable from the pipeline, so kept out of git. Only this README is tracked.

| File | Produced by | Content |
|---|---|---|
| `matches.jsonl` | `limitless_vgc.py` | One row per match, raw |
| `teams.jsonl` | `limitless_vgc.py` | One row per (tournament, player): placing, record, team |
| `matches_clean.jsonl` | `canonicalize.py` | `matches.jsonl` with canonicalised text fields |

## `matches.jsonl` schema

```json
{
  "tournament": "...", "date": "...", "phase": 1, "round": 3,
  "player1": "...", "player2": "...", "winner": "...", "label": 1,
  "team1": [ { "species": "Charizard", "item": "Charizardite Y", "ability": "Blaze",
               "nature": "Modest", "moves": ["Heat Wave","Solar Beam","Weather Ball","Protect"],
               "tera_type": null, "evs": null }, "... x6" ],
  "team2": [ "... x6" ]
}
```

`label` is 1 when player1 won, 0 otherwise; ties excluded at load. A `pairings` row is one
**match** (a best-of-three is a single row with the series winner), consistent with the
Bernoulli likelihood. `tera_type` is always null in Champions. `evs` is not collected — spreads
are often unavailable and were deliberately excluded from the model.

## Regenerating

```bash
python src/limitless_vgc.py --format <FORMAT_ID> --min-players 30 --max-tournaments 300
python src/canonicalize.py data/matches.jsonl data/matches_clean.jsonl --fuzz 85
```

Requires network access to `play.limitlesstcg.com`. Canonicalisation downloads PokeAPI tables
into `pokeapi_csv/` on first run.

## Known quirks

Text fields arrive in the submitting player's game language, with typos — hence
canonicalisation. Two things survive it deliberately: the new Champions **Mega Stones**
(Floettite, Raichunite and 50 others), absent from PokeAPI but internally consistent and kept
as high-value features; and **missing abilities** in ~8,500 slots, recorded as null and handled
by partial pooling rather than row deletion. Use `record`, not `placing`, for any weighting —
`placing` is null for players who drop mid-event.