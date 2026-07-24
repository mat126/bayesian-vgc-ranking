"""
design_matrix.py
Da data/matches.jsonl costruisce gli ingredienti del Bradley-Terry gerarchico.

Per ogni blocco categorico (species, moves, ability, item, nature) una matrice
di DIFFERENZA dei conteggi tra team1 e team2:
    Xb_diff[m, l] = (#volte livello l in team1_m) - (#volte in team2_m)
piu' gli indici giocatore (p1, p2) e l'esito y.

Formulazione del modello a valle:
    eta_m = c_side + sum_b (Xb_diff[m] . beta_b) + lam[p1_m] - lam[p2_m]

Le voci rare si tagliano con una soglia di frequenza per blocco: cosi' i livelli
localizzati rari (nature/mosse/abilita' in altre lingue) e i one-off si ripiegano
sul baseline invece di generare centinaia di categorie viste pochissime volte.

Test offline:  python design_matrix.py data/matches.jsonl
"""
from __future__ import annotations
import json, collections
import numpy as np

SINGLE = ("species", "ability", "item", "nature")   # un valore per Pokemon
ALL_BLOCKS = list(SINGLE) + ["moves"]               # moves e' multi-valore (4 per Pokemon)


def _norm(x):
    return x.strip() if isinstance(x, str) else x


def load_matches(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("label") is None:      # scarta i pareggi
                continue
            out.append(m)
    return out


def team_counts(team):
    c = {b: collections.Counter() for b in ALL_BLOCKS}
    for mon in team:
        for b in SINGLE:
            v = _norm(mon.get(b))
            if v:
                c[b][v] += 1
        for mv in mon.get("moves", []):
            mv = _norm(mv)
            if mv:
                c["moves"][mv] += 1
    return c


def build_vocabs(matches, min_freq):
    """min_freq: dict blocco -> occorrenze minime (contate su entrambi i lati)."""
    tot = {b: collections.Counter() for b in ALL_BLOCKS}
    for m in matches:
        for team in (m["team1"], m["team2"]):
            c = team_counts(team)
            for b in ALL_BLOCKS:
                tot[b].update(c[b])
    vocab = {}
    for b in ALL_BLOCKS:
        thr = min_freq.get(b, 1)
        levels = sorted(lv for lv, n in tot[b].items() if n >= thr)
        vocab[b] = {lv: i for i, lv in enumerate(levels)}
    return vocab, tot


def build_matrices(matches, vocab, min_player_matches=1):
    """min_player_matches: un giocatore riceve un effetto proprio solo se ha almeno
    N match. Gli altri vengono mandati su un indice sentinella (= len(players)) il cui
    effetto e' fissato a 0 nel modello, cioe' vengono poolati sulla media.
    Serve al condizionamento: i giocatori con 2-3 match sono direzioni quasi piatte
    del posterior e costringono NUTS a traiettorie lunghissime."""
    M = len(matches)
    X = {b: np.zeros((M, len(vocab[b])), dtype=np.float32) for b in ALL_BLOCKS}

    freq = collections.Counter()
    for m in matches:
        freq[m["player1"]] += 1
        freq[m["player2"]] += 1
    keep = {p for p, n in freq.items() if n >= min_player_matches}

    players = {}
    def pid(p):
        if p not in keep:
            return -1                      # segnaposto, rimappato a n_players dopo
        return players.setdefault(p, len(players))
    p1 = np.empty(M, dtype=np.int64)
    p2 = np.empty(M, dtype=np.int64)
    y  = np.empty(M, dtype=np.int64)
    for i, m in enumerate(matches):
        c1, c2 = team_counts(m["team1"]), team_counts(m["team2"])
        for b in ALL_BLOCKS:
            vb = vocab[b]
            for lv, n in c1[b].items():
                j = vb.get(lv)
                if j is not None:
                    X[b][i, j] += n
            for lv, n in c2[b].items():
                j = vb.get(lv)
                if j is not None:
                    X[b][i, j] -= n
        p1[i], p2[i] = pid(m["player1"]), pid(m["player2"])
        y[i] = m["label"]
    n = len(players)
    p1[p1 < 0] = n                          # sentinella -> ultimo indice (effetto 0)
    p2[p2 < 0] = n
    inv_players = {v: k for k, v in players.items()}
    return X, p1, p2, y, players, inv_players


DEFAULT_MIN_FREQ = {"species": 1, "moves": 20, "ability": 20, "item": 20, "nature": 20}


def build_side_species(matches, vocab):
    """Matrici di conteggio-specie PER LATO (non differenza): servono al termine di
    sinergia, che e' quadratico e non si riduce alla differenza dei conteggi."""
    vb = vocab["species"]
    M, n = len(matches), len(vb)
    SA = np.zeros((M, n), dtype=np.float32)
    SB = np.zeros((M, n), dtype=np.float32)
    for i, m in enumerate(matches):
        for mon in m["team1"]:
            j = vb.get(mon["species"])
            if j is not None:
                SA[i, j] += 1
        for mon in m["team2"]:
            j = vb.get(mon["species"])
            if j is not None:
                SB[i, j] += 1
    return SA, SB


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/matches.jsonl"
    matches = load_matches(path)
    vocab, tot = build_vocabs(matches, DEFAULT_MIN_FREQ)
    print("match con esito:", len(matches))
    for b in ALL_BLOCKS:
        print(f"  {b:8s}: {len(vocab[b]):4d} livelli tenuti "
              f"(su {len(tot[b])} grezzi, soglia {DEFAULT_MIN_FREQ.get(b,1)})")
    X, p1, p2, y, players, _ = build_matrices(matches, vocab)
    print("  giocatori distinti:", len(players))
    print("  P(vince p1):", round(float(y.mean()), 3))
    print("  shape X_species:", X["species"].shape)