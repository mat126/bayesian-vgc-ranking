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

        # PARAMETRIZZAZIONE NON CENTRATA: la scala tau moltiplica una variabile
        # standardizzata invece di entrare come sigma. ZeroSumNormal e' non-centrata
        # solo rispetto al vincolo somma-zero, NON rispetto a tau: scrivendo
        # sigma=tau si crea l'imbuto gerarchico che blocca il campionatore.
        beta_raw = {b: pm.ZeroSumNormal(f"beta_{b}_raw", sigma=1.0, dims=b) for b in ALL_BLOCKS}
        beta = {b: pm.Deterministic(f"beta_{b}", beta_raw[b] * tau[b], dims=b) for b in ALL_BLOCKS}
        lam_raw = pm.ZeroSumNormal("lam_player_raw", sigma=1.0, dims="player")
        lam = pm.Deterministic("lam_player", lam_raw * tau_player, dims="player")
        # slot extra con effetto 0: ci finiscono i giocatori sotto la soglia di match
        lam_full = pt.concatenate([lam, pt.zeros(1)])

        eta = c_side
        for b in ALL_BLOCKS:
            eta = eta + pm.math.dot(X[b], beta[b])
        eta = eta + lam_full[p1] - lam_full[p2]

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
    ap.add_argument("--sampler", choices=["pymc", "nutpie", "numpyro"], default="nutpie",
                    help="backend NUTS: 'nutpie' (Rust, veloce su CPU), 'numpyro' (JAX, usa GPU), 'pymc' (default interno)")
    ap.add_argument("--tune", type=int, default=1000,
                    help="passi di adattamento del NUTS (500 spesso bastano e dimezzano i tempi)")
    ap.add_argument("--chains", type=int, default=4,
                    help="catene NUTS in parallelo; 2 riduce la contesa di CPU/RAM su portatile")
    ap.add_argument("--target-accept", type=float, default=0.9,
                    help="0.9 e' prudente; con zero divergenze si puo' scendere a 0.8, "
                         "che allunga il passo e dimezza le valutazioni di gradiente per draw")
    ap.add_argument("--min-player-matches", type=int, default=1,
                    help="match minimi perche' un giocatore abbia un effetto proprio (prova 8); "
                         "gli altri sono poolati a 0. Migliora molto il condizionamento.")
    ap.add_argument("--out-dir", default="models",
                    help="cartella dove salvare posterior e artefatti")
    ap.add_argument("--tag", default=None,
                    help="etichetta del run: i file diventano bt_posterior_<tag>.nc e "
                         "bt_artifacts_<tag>.json. Se omessa e' dedotta da metodo e sinergia "
                         "(es. 'nuts_syn0', 'nuts_syn4'), cosi' run diversi non si sovrascrivono.")
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
    X, p1, p2, y, players, inv_players = build_matrices(matches, vocab, a.min_player_matches)
    print("dimensioni blocco:", {b: X[b].shape[1] for b in ALL_BLOCKS},
          "| giocatori con effetto:", len(players),
          f"(soglia {a.min_player_matches} match) | match:", len(matches))

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
            kw = dict(draws=a.draws, tune=a.tune, chains=a.chains, target_accept=a.target_accept)
            if a.sampler == "pymc":
                kw["init"] = "jitter+adapt_diag"
            else:
                kw["nuts_sampler"] = a.sampler      # 'nutpie' oppure 'numpyro'
            print(f"campionatore NUTS: {a.sampler} | tune={a.tune} draws={a.draws} "
                  f"chains={a.chains} target_accept={a.target_accept}")
            idata = pm.sample(**kw)
        else:
            idata = pm.find_MAP()

    if a.method != "map":
        import json, os
        tag = a.tag or f"{a.method}_syn{a.synergy_dim}"
        os.makedirs(a.out_dir, exist_ok=True)
        post_path = os.path.join(a.out_dir, f"bt_posterior_{tag}.nc")
        art_path  = os.path.join(a.out_dir, f"bt_artifacts_{tag}.json")

        idata.to_netcdf(post_path)                  # metodo dell'oggetto (az.to_netcdf rimosso)
        print(f"posterior salvata in {post_path}")
        # artefatti per il notebook: vocabolari, giocatori, configurazione del run
        with open(art_path, "w", encoding="utf-8") as f:
            json.dump({"tag": tag,
                       "config": {"method": a.method,
                                  "sampler": a.sampler if a.method == "nuts" else None,
                                  "synergy_dim": a.synergy_dim,
                                  "min_freq": a.min_freq,
                                  "min_player_matches": a.min_player_matches,
                                  "tune": a.tune if a.method == "nuts" else None,
                                  "draws": a.draws,
                                  "chains": a.chains if a.method == "nuts" else None,
                                  "n_matches": len(matches),
                                  "n_players_effect": len(players),
                                  "limit": a.limit},
                       "min_freq": mf,
                       "vocab": {b: list(vocab[b].keys()) for b in ALL_BLOCKS},
                       "players": player_labels}, f, ensure_ascii=False)
        print(f"artefatti salvati in {art_path}")
        # esempio: forza a posteriori del team del primo match
        s = posterior_team_strength(idata, vocab, matches[0]["team1"])
        print(f"esempio s(team) primo match: media {s.mean():.3f} "
              f"[{np.percentile(s,5):.3f}, {np.percentile(s,95):.3f}]")