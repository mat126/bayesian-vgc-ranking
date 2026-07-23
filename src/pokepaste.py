"""
pokepaste.py
Parser del formato pokepaste / export Showdown per teamlist VGC.

Per ciascun Pokemon estrae: species, item, ability, nature, moves.
(tera_type ed evs vengono letti se presenti ma non usati nel modello a 4 feature.)

Funziona identicamente sul testo restituito dall'API Limitless e sui pokepaste
di Victory Road / vrpastes, quindi e' l'unico componente di parsing che ti serve.

Test offline (nessuna rete):  python pokepaste.py
"""
from __future__ import annotations
import re

GENDERS = {"M", "F"}


def _parse_header(line: str):
    """Prima riga di un blocco: 'Nickname (Species) (M) @ Item' e varianti."""
    item = None
    if "@" in line:
        line, item = line.rsplit("@", 1)
        item = item.strip()
    line = line.strip()

    parens = re.findall(r"\(([^)]*)\)", line)
    forms = [p for p in parens if p not in GENDERS]   # specie tra parentesi (se c'e' un nickname)
    if forms:
        species = forms[-1].strip()                   # 'Nickname (Species)'
    else:
        species = re.sub(r"\([^)]*\)", "", line).strip()  # nessun nickname: rimuovi solo il gender
    return species, item


def parse_mon(block: str) -> dict:
    lines = [l.rstrip() for l in block.splitlines() if l.strip()]
    species = item = ability = nature = tera = evs = None
    moves = []
    if lines:
        species, item = _parse_header(lines[0])
    for l in lines[1:]:
        s = l.strip()
        low = s.lower()
        if s.startswith("- "):
            moves.append(s[2:].strip())
        elif low.startswith("ability:"):
            ability = s.split(":", 1)[1].strip()
        elif low.startswith("tera type:"):
            tera = s.split(":", 1)[1].strip()
        elif low.startswith("evs:"):
            evs = s.split(":", 1)[1].strip()
        elif s.endswith("Nature"):
            nature = s.replace("Nature", "").strip()
    return {"species": species, "item": item, "ability": ability,
            "nature": nature, "moves": moves, "tera_type": tera, "evs": evs}


def parse_team(text: str) -> list[dict]:
    """Team pokepaste -> lista di dict (uno per Pokemon). I mon sono separati da riga vuota."""
    blocks = re.split(r"\n\s*\n", text.strip())
    team = [parse_mon(b) for b in blocks if b.strip()]
    return [m for m in team if m.get("species")]


def team_to_features(team, move_vocab=None) -> list[dict]:
    """Rappresentazione a 4 feature: multi-hot mosse (opz. ristrette a mosse trasferibili),
    piu' ability/item/nature/species. Species e' tenuta a parte come ancora del modello."""
    out = []
    for m in team:
        moves = m["moves"]
        if move_vocab is not None:
            moves = [mv for mv in moves if mv in move_vocab]
        out.append({"species": m["species"], "ability": m["ability"],
                    "item": m["item"], "nature": m["nature"], "moves": sorted(moves)})
    return out


if __name__ == "__main__":
    sample = """\
Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Fake Out
- Knock Off
- Parting Shot
- Will-O-Wisp

Amoonguss @ Rocky Helmet
Ability: Regenerator
Level: 50
EVs: 236 HP / 156 Def / 116 SpD
Calm Nature
- Spore
- Rage Powder
- Pollen Puff
- Protect
"""
    import json
    team = parse_team(sample)
    print(json.dumps(team, ensure_ascii=False, indent=2))
    print("\n--- 4 feature ---")
    print(json.dumps(team_to_features(team), ensure_ascii=False, indent=2))
