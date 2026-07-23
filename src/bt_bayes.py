"""
bt_bayes.py
Bradley-Terry gerarchico bayesiano (PyMC) sugli output di design_matrix.py.

  eta_m = c_side
        + sum_b Xb_diff[m] . beta_b        (species, moves, ability, item, nature)
        + lam[p1_m] - lam[p2_m]            (abilita' del pilota)
  y_m ~ Bernoulli(sigmoid(eta_m))

Ogni blocco ha una scala gerarchica propria (tau_b) e vincolo somma-zero via
ZeroSumNormal, che risolve l'identificabilita' (la costante per blocco si cancella
nella differenza) e dà lo shrinkage sui livelli rari. ZeroSumNormal e' gia'
non-centrata internamente, quindi campiona bene.

  python bt_bayes.py --method advi    # veloce, sanity check
  python bt_bayes.py --method nuts    # lento, stime finali
"""
from __future__ import annotations
import argparse
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from design_matrix import (load_matches, build_vocabs, build_matrices, build_side_species,
                           team_counts, ALL_BLOCKS, DEFAULT_MIN_FREQ)


def make_model(X, p1, p2, y, labels, player_labels, SA=None, SB=None, synergy_dim=0):
    coords = {b: labels[b] for b in ALL_BLOCKS}
    coords["player"] = player_labels
    if synergy_dim:
        coords["syn"] = list(range(synergy_dim))
    with pm.Model(coords=coords) as model:
        c_side = pm.Normal("c_side", 0.0, 0.5)                       # intercetta di lato

        tau = {b: pm.HalfNormal(f"tau_{b}", 1.0) for b in ALL_BLOCKS}
        tau_player = pm.HalfNormal("tau_player", 1.0)

        beta = {b: pm.ZeroSumNormal(f"beta_{b}", sigma=tau[b], dims=b) for b in ALL_BLOCKS}
        lam = pm.ZeroSumNormal("lam_player", sigma=tau_player, dims="player")

        eta = c_side
        for b in ALL_BLOCKS:
            eta = eta + pm.math.dot(X[b], beta[b])
        eta = eta + lam[p1] - lam[p2]

        # sinergia a rango basso specie x specie (Factorization Machine)
        if synergy_dim and SA is not None:
            tau_syn = pm.HalfNormal("tau_syn", 0.2)               # prior stretto: limita l'esplosione quadratica
            Zv = pm.Normal("Zv_species", 0.0, 1.0, dims=("species", "syn"))
            V = pm.Deterministic("V_species", Zv * tau_syn, dims=("species", "syn"))  # non-centrata
            h = (V ** 2).sum(axis=1)                     # norma^2 per specie
            ZA = pt.dot(SA, V); ZB = pt.dot(SB, V)       # somma dei vettori del team, per lato
            psiA = 0.5 * ((ZA ** 2).sum(axis=1) - pt.dot(SA, h))
            psiB = 0.5 * ((ZB ** 2).sum(axis=1) - pt.dot(SB, h))
            eta = eta + (psiA - psiB)

        eta = pt.clip(eta, -30.0, 30.0)                  # stabilita' numerica: evita overflow/NaN
        pm.Bernoulli("y", logit_p=eta, observed=y)
    return model


def posterior_team_strength(idata, vocab, team):
    """Distribuzione a posteriori della forza s(team) (senza pilota), per il
    ranking 'chi e' favorito'. team = lista di dict come in matches.jsonl."""
    import xarray as xr
    post = idata.posterior
    c = team_counts(team)
    s = post["c_side"] * 0.0                    # zeros con dims (chain, draw)
    for b in ALL_BLOCKS:
        vb = vocab[b]
        vec = np.zeros(len(vb), dtype=np.float32)
        for lv, n in c[b].items():
            j = vb.get(lv)
            if j is not None:
                vec[j] += n
        beta = post[f"beta_{b}"]                # (chain, draw, level)
        vec_da = xr.DataArray(vec, dims=[b], coords={b: list(vb.keys())})
        s = s + xr.dot(beta, vec_da, dims=[b])

    # sinergia, se il modello e' stato stimato con --synergy-dim > 0
    if "V_species" in post:
        V = post["V_species"]                   # (chain, draw, species, syn)
        vb = vocab["species"]
        sc = np.zeros(len(vb), dtype=np.float32)
        for lv, n in c["species"].items():
            j = vb.get(lv)
            if j is not None:
                sc[j] += n
        sc_da = xr.DataArray(sc, dims=["species"], coords={"species": list(vb.keys())})
        z = xr.dot(V, sc_da, dims=["species"])          # (chain, draw, syn)
        hnorm = (V ** 2).sum("syn")                     # (chain, draw, species)
        self_term = xr.dot(hnorm, sc_da, dims=["species"])
        s = s + 0.5 * ((z ** 2).sum("syn") - self_term)

    return s.stack(sample=("chain", "draw")).values   # array di draw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="data/matches.jsonl")
    ap.add_argument("--method", choices=["advi", "nuts", "map"], default="advi")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--advi-iter", type=int, default=30000)
    ap.add_argument("--min-freq", type=int, default=20,
                    help="soglia per moves/ability/item/nature (species resta 1)")
    ap.add_argument("--limit", type=int, default=0,
                    help="usa solo N match (campione casuale, seed fisso); 0 = tutti. Per test veloci.")
    ap.add_argument("--synergy-dim", type=int, default=0,
                    help="dimensione dei vettori latenti per la sinergia specie x specie (0 = off; prova 4)")
    a = ap.parse_args()

    matches = load_matches(a.matches)
    if a.limit and a.limit < len(matches):
        import random
        random.seed(0)
        matches = random.sample(matches, a.limit)
        print(f"[test] campione ridotto a {len(matches)} match")
    mf = dict(DEFAULT_MIN_FREQ)
    for b in ("moves", "ability", "item", "nature"):
        mf[b] = a.min_freq
    vocab, tot = build_vocabs(matches, mf)
    X, p1, p2, y, players, inv_players = build_matrices(matches, vocab)
    print("dimensioni blocco:", {b: X[b].shape[1] for b in ALL_BLOCKS},
          "| giocatori:", len(players), "| match:", len(matches))

    labels = {b: list(vocab[b].keys()) for b in ALL_BLOCKS}
    player_labels = [inv_players[i] for i in range(len(players))]

    SA = SB = None
    if a.synergy_dim:
        SA, SB = build_side_species(matches, vocab)
        print(f"sinergia attiva: {SA.shape[1]} specie x {a.synergy_dim} dimensioni latenti")

    model = make_model(X, p1, p2, y, labels, player_labels, SA, SB, a.synergy_dim)
    with model:
        if a.method == "advi":
            if a.synergy_dim:
                # ADVI stabilizzato per il termine quadratico: learning rate piu' basso
                approx = pm.fit(a.advi_iter, method="advi",
                                obj_optimizer=pm.adagrad_window(learning_rate=1e-3))
            else:
                approx = pm.fit(a.advi_iter, method="advi")
            idata = approx.sample(a.draws)
        elif a.method == "nuts":
            idata = pm.sample(a.draws, tune=1000, target_accept=0.9,
                              chains=4, init="jitter+adapt_diag")
        else:
            idata = pm.find_MAP()

    if a.method != "map":
        idata.to_netcdf("bt_posterior.nc")          # metodo dell'oggetto (az.to_netcdf rimosso)
        print("posterior salvata in bt_posterior.nc")
        # artefatti per il notebook: vocabolari, giocatori, soglie
        import json
        with open("bt_artifacts.json", "w", encoding="utf-8") as f:
            json.dump({"min_freq": mf,
                       "vocab": {b: list(vocab[b].keys()) for b in ALL_BLOCKS},
                       "players": player_labels}, f, ensure_ascii=False)
        print("artefatti salvati in bt_artifacts.json")
        # esempio: forza a posteriori del team del primo match
        s = posterior_team_strength(idata, vocab, matches[0]["team1"])
        print(f"esempio s(team) primo match: media {s.mean():.3f} "
              f"[{np.percentile(s,5):.3f}, {np.percentile(s,95):.3f}]")