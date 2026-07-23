"""
inspect_standings.py
Diagnostica: mostra la struttura REALE di uno standing Limitless, cosi' sappiamo
esattamente dove sta la teamlist e come si chiama il campo.

Non serve nessun format id: trova da solo un torneo VGC con decklists.
Uso:  python inspect_standings.py
"""
import json
import limitless_vgc as L

# tornei VGC piu' recenti (nessun filtro di formato necessario per la diagnostica)
tournaments = L.get_tournaments(limit=25)

chosen = None
for t in tournaments:
    det = L.get_details(t["id"])
    if det.get("decklists"):
        chosen = t
        break

if not chosen:
    print("Nessun torneo con decklists nei primi 25. Rilancia alzando limit.")
    raise SystemExit

print(f"Torneo scelto: {chosen['name']}  (id {chosen['id']})\n")

st = L.get_standings(chosen["id"])
print("Numero di giocatori nello standing:", len(st))
print("KEYS del primo giocatore:", list(st[0].keys()))
print("\n--- primo giocatore (troncato) ---")
print(json.dumps(st[0], ensure_ascii=False, indent=2)[:3000])

# se una chiave contiene un oggetto (non stringa), mostRiamone i sotto-campi
for k, v in st[0].items():
    if isinstance(v, (dict, list)) and v:
        print(f"\n--- dettaglio campo '{k}' ({type(v).__name__}) ---")
        print(json.dumps(v, ensure_ascii=False, indent=2)[:1500])
