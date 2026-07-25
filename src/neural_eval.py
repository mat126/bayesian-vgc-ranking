"""
neural_eval.py
Valuta la rete addestrata e produce i risultati, confrontabili col modello bayesiano.

  python neural_eval.py --ckpt neural_ckpt.pt --matches data/matches_clean.jsonl --teams data/teams.jsonl

Fa tre cose:
  1) metriche held-out sullo STESSO split del training (log-loss, accuracy, calibrazione)
  2) classifica delle squadre per forza s(team), con incertezza via MC-dropout
  3) testa a testa tra le prime due
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from neural_data import load_matches, split_matches, VGCDataset, encode_team, FIELDS
from neural_bt import TeamBT


def load_model(ckpt_path, device):
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    model = TeamBT(ck["sizes"], d=ck["config"]["d"], h=ck["config"]["h"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def team_tensor(team, vocab, device):
    enc = encode_team(team, vocab)
    return {f: torch.tensor([enc[f]], dtype=torch.long, device=device) for f in FIELDS}


@torch.no_grad()
def held_out_metrics(model, val_matches, vocab, device, out_dir):
    from torch.utils.data import DataLoader
    dl = DataLoader(VGCDataset(val_matches, vocab, augment=False), batch_size=512)
    ps, ys = [], []
    for A, B, y in dl:
        A = {k: v.to(device) for k, v in A.items()}
        B = {k: v.to(device) for k, v in B.items()}
        ps.append(torch.sigmoid(model(A, B)).cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    eps = 1e-9
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    acc = np.mean((p > 0.5) == y)
    print(f"\nHELD-OUT (val)  log-loss {ll:.4f}  (coin-flip {np.log(2):.4f})  |  accuracy {acc:.4f}")

    # calibrazione
    try:
        import matplotlib.pyplot as plt
        bins = np.quantile(p, np.linspace(0, 1, 11)); bins[0], bins[-1] = 0, 1
        mid, obs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p >= lo) & (p < hi)
            if sel.sum():
                mid.append(p[sel].mean()); obs.append(y[sel].mean())
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(mid, obs, "o-", color="#e8590c")
        ax.set_xlabel("probabilita' prevista"); ax.set_ylabel("frequenza osservata")
        ax.set_title("Calibrazione rete (held-out)")
        os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout(); fig.savefig(f"{out_dir}/calibrazione_rete.png", dpi=130)
        print(f"calibrazione salvata in {out_dir}/calibrazione_rete.png")
    except Exception as e:
        print("grafico calibrazione saltato:", e)
    return ll, acc


def mc_scores(model, team_t, n_mc):
    """MC-dropout: attiva il dropout e campiona n forze -> distribuzione (incertezza)."""
    model.train()                       # dropout attivo
    with torch.no_grad():
        s = np.array([model.score(team_t).item() for _ in range(n_mc)])
    model.eval()
    return s


def team_key(team):
    return tuple(sorted((m["species"], m.get("item"), m.get("ability"), m.get("nature"),
                         tuple(sorted(m.get("moves", [])))) for m in team))


def rank_teams(model, teams_path, vocab, device, min_occ, n_mc, out_dir):
    rows = [json.loads(l) for l in open(teams_path, encoding="utf-8")]
    uniq = {}
    for r in rows:
        d = uniq.setdefault(team_key(r["team"]), {"team": r["team"], "n": 0})
        d["n"] += 1
    cand = [v for v in uniq.values() if v["n"] >= min_occ]
    print(f"\n{len(cand)} squadre con almeno {min_occ} presenze (su {len(uniq)} uniche)")

    recs = []
    for v in cand:
        s = mc_scores(model, team_tensor(v["team"], vocab, device), n_mc)
        recs.append({"squadra": " / ".join(m["species"] for m in v["team"]),
                     "n": v["n"], "mean": float(s.mean()),
                     "q5": float(np.percentile(s, 5)), "q95": float(np.percentile(s, 95)),
                     "_team": v["team"]})
    recs.sort(key=lambda r: -r["mean"])

    import csv
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/ranking_squadre_rete.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["squadra", "n", "mean", "q5", "q95"])
        for r in recs:
            w.writerow([r["squadra"], r["n"], round(r["mean"], 4), round(r["q5"], 4), round(r["q95"], 4)])
    print("== Top 20 squadre (forza s, MC-dropout) ==")
    for r in recs[:20]:
        print(f"  {r['mean']:+.3f}  [{r['q5']:+.3f},{r['q95']:+.3f}]  n={r['n']:3d}  {r['squadra']}")
    print(f"ranking salvato in {out_dir}/ranking_squadre_rete.csv")
    return recs


def head_to_head(model, tA, tB, vocab, device, n_mc):
    a, b = team_tensor(tA, vocab, device), team_tensor(tB, vocab, device)
    model.train()
    with torch.no_grad():
        p = np.array([torch.sigmoid(model.score(a) - model.score(b)).item() for _ in range(n_mc)])
    model.eval()
    print(f"\nTesta a testa:\n  A: {' / '.join(m['species'] for m in tA)}"
          f"\n  B: {' / '.join(m['species'] for m in tB)}"
          f"\n  P(A batte B) = {p.mean():.3f}  [{np.percentile(p,5):.3f}, {np.percentile(p,95):.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="neural_ckpt.pt")
    ap.add_argument("--matches", default="data/matches_clean.jsonl")
    ap.add_argument("--teams", default="data/teams.jsonl")
    ap.add_argument("--min-occ", type=int, default=5)
    ap.add_argument("--mc", type=int, default=50, help="campioni MC-dropout per l'incertezza")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.ckpt, device)
    vocab = ck["vocab"]
    print("device:", device, "| val log-loss al training:", round(ck.get("best_val_logloss", float('nan')), 4))

    # 1) metriche held-out sullo stesso split del training
    _, val = split_matches(load_matches(a.matches), ck["val_frac"], ck["seed"])
    held_out_metrics(model, val, vocab, device, a.out_dir)

    # 2) ranking squadre
    recs = rank_teams(model, a.teams, vocab, device, a.min_occ, a.mc, a.out_dir)

    # 3) testa a testa tra le prime due
    if len(recs) >= 2:
        head_to_head(model, recs[0]["_team"], recs[1]["_team"], vocab, device, a.mc)


if __name__ == "__main__":
    main()
