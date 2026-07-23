"""
showdown_replays.py  --  FONTE SECONDARIA (ladder, non tornei)

Scarica replay pubblici di Pokemon Showdown e ne estrae, dal battle log,
la composizione a livello SPECIE dei due team + il vincitore.

QUANDO USARLO
- Volume extra di match a livello specie (utile per il BT su specie/archetipo).
- NON per il modello a 4 feature: sul ladder mosse/abilita'/strumento/natura
  spesso non sono tutte rivelate nel log. Per i set completi usa Limitless.
- E' dato di ladder: distribuzione diversa dal meta da torneo. Pesalo a parte
  (o usalo solo come prior), non mescolarlo 1:1 con i match da torneo.

DA VERIFICARE prima dell'uso
- Il format id corrente (cerca su Showdown, es. 'gen9vgc2026regg' o simile).
- Gli endpoint pubblici seguono questo schema noto ma vanno confermati:
    lista replay: https://replay.pokemonshowdown.com/search.json?format=<FMT>&page=<n>
    singolo log:  https://replay.pokemonshowdown.com/<id>.json   -> {"log": "...", ...}
- Per un parser COMPLETO del battle log (picks, leads, mosse, tera, winner)
  conviene riusare VS Recorder: github.com/Pocolip/vs-recorder

Uso:  python showdown_replays.py --format gen9vgc2026regg --pages 5 --out data/showdown.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time
import requests

BASE = "https://replay.pokemonshowdown.com"
S = requests.Session()
S.headers.update({"User-Agent": "vgc-bt-research/0.1"})


def list_replays(fmt, pages=5):
    ids = []
    for p in range(1, pages + 1):
        r = S.get(f"{BASE}/search.json", params={"format": fmt, "page": p}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        ids += [d["id"] for d in data]
        time.sleep(0.5)
    return ids


def fetch_log(replay_id):
    r = S.get(f"{BASE}/{replay_id}.json", timeout=30)
    r.raise_for_status()
    return r.json()


def parse_species_and_winner(log_json) -> dict:
    """Le righe |poke|pX|Species,... elencano il team di ogni lato (sempre presente);
    |win|Name da' il vincitore. Affidabile a livello SPECIE."""
    log = log_json.get("log", "")
    p1 = p2 = winner = None
    t1, t2 = [], []
    for line in log.splitlines():
        parts = line.split("|")
        if line.startswith("|player|p1|"):
            p1 = parts[3]
        elif line.startswith("|player|p2|"):
            p2 = parts[3]
        elif line.startswith("|poke|p1|"):
            t1.append(parts[3].split(",")[0])
        elif line.startswith("|poke|p2|"):
            t2.append(parts[3].split(",")[0])
        elif line.startswith("|win|"):
            winner = line.split("|", 2)[2].strip()
    side = 1 if winner and winner == p1 else (2 if winner and winner == p2 else 0)
    return {"id": log_json.get("id"), "format": log_json.get("format"),
            "p1": p1, "p2": p2, "team1": t1, "team2": t2,
            "winner_name": winner, "winner_side": side}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, help="es. gen9vgc2026regg (verifica sul sito)")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--out", default="data/showdown.jsonl")
    a = ap.parse_args()

    ids = list_replays(a.format, a.pages)
    print(f"{len(ids)} replay trovati", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, rid in enumerate(ids, 1):
            try:
                rec = parse_species_and_winner(fetch_log(rid))
            except Exception as e:
                print(f"skip {rid}: {e}", file=sys.stderr); continue
            if rec["winner_side"] == 0 or not rec["team1"] or not rec["team2"]:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(ids)}", file=sys.stderr)
            time.sleep(0.4)
    print(f"scritto {a.out}", file=sys.stderr)
