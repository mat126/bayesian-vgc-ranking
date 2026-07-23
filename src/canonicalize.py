"""
canonicalize.py
Pulisce i campi testuali sporchi (nature, mosse, abilita', strumenti) mappandoli
alla forma inglese CANONICA. Il vocabolario e' chiuso e noto, quindi e' quasi tutto
automatico.

  slug (minuscole, via accenti/punteggiatura)      -> case, spazi, trattini, apostrofi
  token nulli per-campo (None/none/No Item/...)     -> item: 'No Item'; altri: mancante
  aggancio esatto alla tabella multilingua PokeAPI  -> tutte le lingue ufficiali
  fuzzy match sul residuo (rapidfuzz)               -> i typo
  flag                                              -> resta un pugno di casi da guardare

Gli item nuovi di Champions non presenti in PokeAPI (Mega Stone tipo 'Floettite')
NON vengono forzati: restano col loro nome grezzo, che e' pulito e consistente, e
funzionano gia' come feature. Il report li elenca a parte come "nuovi item (tenuti)".

Species NON viene toccata: da Limitless arriva gia' pulita in inglese.

Uso:
  pip install rapidfuzz requests   (o via conda-forge)
  python canonicalize.py data/matches.jsonl data/matches_clean.jsonl --fuzz 85
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, unicodedata

CSV_BASE = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv"
FIELDS = {
    "nature":  "nature_names.csv",
    "moves":   "move_names.csv",
    "ability": "ability_names.csv",
    "item":    "item_names.csv",
}
EN_LANG = 9
NULL_SLUGS = {"none", "noitem", "noability", "nil", "na", "null", "", "nothing"}  # "vuoti"
NEW_ITEM_RE = re.compile(r"ite[xy]?$")   # euristica SOLO per il report: Mega Stone nuove

# Override manuali per i residui: chiave = slug(grezzo), valore = forma canonica
# oppure _DROP per i casi di campo-sbagliato/spazzatura (trattati come mancanti).
# Aggiungere righe qui e' preferibile a correggere il dato a mano: e' riproducibile.
_DROP = "__drop__"
MANUAL = {
    "nature": {
        "quite": "Quiet", "adament": "Adamant", "adman": "Adamant",
        "adanant": "Adamant", "modesh": "Modest", "modesy": "Modest",
        "sasay": "Sassy",
        "protean": _DROP, "blaze": _DROP,        # abilita' nel campo natura
    },
    "moves": {
        "physicfangs": "Psychic Fangs", "tailwing": "Tailwind",
        "knockiff": "Knock Off", "overhear": "Overheat",
        "kowtowcleaf": "Kowtow Cleave", "kowtonkleave": "Kowtow Cleave",
        "kowtow": "Kowtow Cleave", "sunday": "Sunny Day",
        "protetc": "Protect", "protech": "Protect", "heatwafe": "Heat Wave",
        "watercrash": "Wave Crash", "lastreason": "Last Respects",
        "fireblitz": "Flare Blitz",
        # "bodyfang", "gmaxgoldrush": ambigui -> lasciati irrisolti di proposito
    },
    "ability": {
        "cursedbodylevel": "Cursed Body", "cyrsebody": "Cursed Body",
        "unberden": "Unburden", "unnvern": "Unnerve", "rough": "Rough Skin",
        "toxictouch": "Poison Touch", "toughclearbody": "Clear Body",
        "sassy": _DROP, "noability": _DROP,       # natura nel campo abilita' / vuoto
    },
    "item": {
        "fokusslash": "Focus Sash",
        "male": _DROP, "female": _DROP,           # genere finito nel campo strumento
        "n": _DROP, "unremarkableform": _DROP,    # spazzatura
    },
}


def slug(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def download_refs(dst="pokeapi_csv"):
    import requests
    os.makedirs(dst, exist_ok=True)
    for fname in FIELDS.values():
        path = os.path.join(dst, fname)
        if os.path.exists(path):
            continue
        r = requests.get(f"{CSV_BASE}/{fname}", timeout=30)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(r.text)
        print(f"scaricato {fname}", file=sys.stderr)


def load_maps(dst="pokeapi_csv"):
    maps = {}
    for field, fname in FIELDS.items():
        rows = list(csv.DictReader(open(os.path.join(dst, fname), encoding="utf-8")))
        idcol = list(rows[0].keys())[0]
        by_id = {}
        for r in rows:
            by_id.setdefault(r[idcol], {})[int(r["local_language_id"])] = r["name"]
        exact, canon = {}, {}
        for _id, langs in by_id.items():
            en = langs.get(EN_LANG) or next(iter(langs.values()))
            canon[slug(en)] = en
            for nm in langs.values():
                exact[slug(nm)] = en
        maps[field] = {"exact": exact, "canon": canon}
    return maps


def make_canonicalizer(maps, fuzz_threshold=85):
    try:
        from rapidfuzz import process, fuzz
        have_fuzz = True
    except Exception:
        have_fuzz = False
        print("rapidfuzz assente: niente fuzzy, i typo verranno flaggati", file=sys.stderr)

    def canon(field, raw):
        s = slug(raw)
        man = MANUAL.get(field, {}).get(s)
        if man is not None:
            return (None, "manual-drop") if man == _DROP else (man, "manual")
        if s in NULL_SLUGS:
            return ("No Item", "null") if field == "item" else (None, "missing")
        m = maps[field]
        if s in m["exact"]:
            return m["exact"][s], "exact"
        if have_fuzz:
            hit = process.extractOne(s, list(m["canon"].keys()), scorer=fuzz.ratio)
            if hit and hit[1] >= fuzz_threshold:
                return m["canon"][hit[0]], f"fuzzy:{int(hit[1])}"
        return None, "unresolved"

    return canon


def canon_team(team, canon, unresolved, missing):
    for mon in team:
        for field in ("nature", "ability", "item"):
            val = mon.get(field)
            if val is None:
                missing[field] = missing.get(field, 0) + 1
                continue
            c, how = canon(field, val)
            if c is not None:
                mon[field] = c
            elif how in ("missing", "manual-drop"):
                mon[field] = None
                missing[field] = missing.get(field, 0) + 1
            else:
                unresolved[field][val] = unresolved[field].get(val, 0) + 1
        new_moves = []
        for mv in mon.get("moves", []):
            c, how = canon("moves", mv)
            if c is not None:
                new_moves.append(c)
            elif how in ("missing", "manual-drop"):
                missing["moves"] = missing.get("moves", 0) + 1
            else:
                unresolved["moves"][mv] = unresolved["moves"].get(mv, 0) + 1
                new_moves.append(mv)   # tenuto grezzo: la soglia di frequenza lo ripieghera'
        mon["moves"] = new_moves
    return team


def main(in_path, out_path, dst="pokeapi_csv", fuzz_threshold=85):
    download_refs(dst)
    canon = make_canonicalizer(load_maps(dst), fuzz_threshold)
    unresolved = {f: {} for f in FIELDS}
    missing = {}
    distinct = {f: set() for f in FIELDS}

    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            m = json.loads(line)
            for tk in ("team1", "team2"):
                m[tk] = canon_team(m[tk], canon, unresolved, missing)
                for mon in m[tk]:
                    for f in ("nature", "ability", "item"):
                        if mon.get(f):
                            distinct[f].add(mon[f])
                    distinct["moves"].update(mon.get("moves", []))
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            n += 1

    print(f"\n{n} match scritti in {out_path}")
    print("distinti dopo pulizia:", {f: len(distinct[f]) for f in FIELDS})
    print("valori mancanti (None):", missing)
    for f in FIELDS:
        if not unresolved[f]:
            continue
        toks = unresolved[f]
        new_items = {t: c for t, c in toks.items() if f == "item" and NEW_ITEM_RE.search(slug(t))}
        typos = {t: c for t, c in toks.items() if t not in new_items}
        if new_items:
            tot = sum(new_items.values())
            print(f"\n[{f}] nuovi item tenuti ({len(new_items)} tipi, {tot} occorrenze): "
                  + ", ".join(sorted(new_items, key=lambda x: -new_items[x])[:8]) + " ...")
        if typos:
            top = sorted(typos.items(), key=lambda x: -x[1])[:15]
            print(f"[{f}] da rivedere ({len(typos)} token, tutti bassa freq -> li droppa la soglia):")
            for tok, c in top:
                print(f"   {c:5d}  {tok!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path", nargs="?", default="data/matches.jsonl")
    ap.add_argument("out_path", nargs="?", default="data/matches_clean.jsonl")
    ap.add_argument("--fuzz", type=int, default=85, help="soglia fuzzy (85 consigliata)")
    a = ap.parse_args()
    main(a.in_path, a.out_path, fuzz_threshold=a.fuzz)