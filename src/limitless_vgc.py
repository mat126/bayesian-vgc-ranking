"""
limitless_vgc.py
Client per l'API pubblica di Limitless (https://play.limitlesstcg.com/api).
Costruisce un dataset match-level VGC per il modello Bradley-Terry:
ogni riga = un match giocato con la teamlist parsata di ENTRAMBI i giocatori.

Nessuna API key necessaria per: /tournaments, /details, /standings, /pairings.
Il client e' volutamente lento e gestisce il 429 (rate limit).

FLUSSO CONSIGLIATO
  1) python limitless_vgc.py --list-formats
        -> trova l'id di formato per "Regulation Set M-B" (Mondiali 2026).
  2) python limitless_vgc.py --inspect <TOURNAMENT_ID>
        -> stampa la decklist grezza del 1o giocatore, per confermare che il
           campo 'decklist' sia testo pokepaste (probabile) o un oggetto.
  3) python limitless_vgc.py --format <FORMAT_ID> --min-players 30 --max-tournaments 200
        -> scrive data/matches.jsonl  e  data/teams.jsonl

OUTPUT
  data/matches.jsonl : una riga per match; campi team1/team2 = team parsati,
                       'label' = 1 se ha vinto player1, 0 se player2 (solo match decisivi).
  data/teams.jsonl   : una riga per (torneo, giocatore) con placing/record e team.

NOTA AMBIENTE: se giri questo dentro una sandbox con egress ristretto, devi
aggiungere play.limitlesstcg.com alla allowlist di rete. Da una connessione
normale funziona senza configurazione.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import requests
from pokepaste import parse_team

API = "https://play.limitlesstcg.com/api"
S = requests.Session()
S.headers.update({"User-Agent": "vgc-bt-research/0.1 (contact: mmorella9@gmail.com)"})
PACING = 0.6  # secondi tra richieste, per rispettare il rate limit


def _get(path, params=None, max_retries=5):
    url = f"{API}{path}"
    for _ in range(max_retries):
        r = S.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 5)))
            continue
        r.raise_for_status()
        time.sleep(PACING)
        return r.json()
    r.raise_for_status()


def list_formats():
    return _get("/games")


def get_tournaments(fmt=None, limit=50, page=1):
    params = {"game": "VGC", "limit": limit, "page": page}
    if fmt:
        params["format"] = fmt
    return _get("/tournaments", params)


def get_details(tid):   return _get(f"/tournaments/{tid}/details")
def get_standings(tid): return _get(f"/tournaments/{tid}/standings")
def get_pairings(tid):  return _get(f"/tournaments/{tid}/pairings")


def structured_team(decklist):
    """L'API VGC restituisce 'decklist' gia' come lista di 6 dict:
    {id, name, item, ability, attacks[], nature, tera}. Mappiamo al nostro schema.
    species = 'name' (le Mega restano la specie base, la Mega Stone e' in 'item')."""
    team = []
    for mon in decklist:
        if not isinstance(mon, dict):
            continue
        species = mon.get("name") or mon.get("id")
        if not species:
            continue
        team.append({
            "species": species,
            "item": mon.get("item"),
            "ability": mon.get("ability"),
            "nature": mon.get("nature"),
            "moves": [m for m in (mon.get("attacks") or []) if m],
            "tera_type": mon.get("tera"),
            "evs": None,
        })
    return team


def to_team(decklist):
    """Dispatcher: lista strutturata (API Limitless) o testo pokepaste (vrpastes)."""
    if isinstance(decklist, list):
        return structured_team(decklist)
    if isinstance(decklist, str) and decklist.strip():
        return parse_team(decklist)   # fallback per pastes di Victory Road / vrpastes
    return []


def build(fmt, min_players, max_tournaments, out_dir, include_ties=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fm = (out / "matches.jsonl").open("w", encoding="utf-8")
    ft = (out / "teams.jsonl").open("w", encoding="utf-8")

    kept = 0; page = 1; n_match = 0
    while kept < max_tournaments:
        batch = get_tournaments(fmt=fmt, limit=50, page=page)
        if not batch:
            break
        for t in batch:
            if kept >= max_tournaments:
                break
            if t.get("players", 0) < min_players:
                continue
            tid = t["id"]
            det = get_details(tid)
            if not det.get("decklists"):        # senza teamlist non c'e' segnale a 4 feature
                continue

            standings = get_standings(tid)
            teams = {}
            for s in standings:
                team = to_team(s.get("decklist"))
                if len(team) < 6:          # teamlist assente o incompleta
                    continue
                teams[s["player"]] = team
                ft.write(json.dumps({
                    "tournament": tid, "date": t.get("date"), "format": t.get("format"),
                    "player": s["player"], "placing": s.get("placing"),
                    "record": s.get("record"), "team": team,
                }, ensure_ascii=False) + "\n")

            pairings = get_pairings(tid)
            for m in pairings:
                p1, p2, w = m.get("player1"), m.get("player2"), m.get("winner")
                if not p1 or not p2:            # bye / no-show
                    continue
                if w in (0, "0"):               # pareggio
                    if not include_ties:
                        continue
                    label = None
                elif w in (-1, "-1"):           # doppia sconfitta
                    continue
                else:
                    label = 1 if w == p1 else 0
                if p1 not in teams or p2 not in teams:
                    continue                    # team ignoto -> scarta (modello a 4 feature)
                fm.write(json.dumps({
                    "tournament": tid, "date": t.get("date"), "format": t.get("format"),
                    "phase": m.get("phase"), "round": m.get("round"),
                    "player1": p1, "player2": p2, "winner": w, "label": label,
                    "team1": teams[p1], "team2": teams[p2],
                }, ensure_ascii=False) + "\n")
                n_match += 1

            kept += 1
            print(f"[{kept}] {t.get('name')} ({t.get('players')}p) "
                  f"-> {len(teams)} team, {n_match} match totali", file=sys.stderr)
        page += 1

    fm.close(); ft.close()
    print(f"\nFatto. {kept} tornei, {n_match} match in {out}/matches.jsonl", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-formats", action="store_true", help="stampa i formati disponibili")
    ap.add_argument("--inspect", metavar="TID", help="stampa la decklist grezza del 1o giocatore di un torneo")
    ap.add_argument("--format", default=None, help="format id (vedi --list-formats), es. Reg M-B")
    ap.add_argument("--min-players", type=int, default=20)
    ap.add_argument("--max-tournaments", type=int, default=100)
    ap.add_argument("--out", default="data")
    ap.add_argument("--include-ties", action="store_true")
    a = ap.parse_args()

    if a.list_formats:
        print(json.dumps(list_formats(), ensure_ascii=False, indent=2)); sys.exit()
    if a.inspect:
        st = get_standings(a.inspect)
        first = next((s for s in st if s.get("decklist")), None)
        print(json.dumps(first, ensure_ascii=False, indent=2)); sys.exit()

    build(a.format, a.min_players, a.max_tournaments, a.out, a.include_ties)
