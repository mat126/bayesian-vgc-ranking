import json, collections
labels, sp, mv, it, ab, na = [], set(), set(), set(), set(), set()
for line in open("data/matches.jsonl", encoding="utf-8"):
    m = json.loads(line)
    if m["label"] is not None: labels.append(m["label"])
    for t in (m["team1"], m["team2"]):
        for mon in t:
            sp.add(mon["species"]); it.add(mon["item"]); ab.add(mon["ability"]); na.add(mon["nature"])
            mv.update(mon["moves"])
print("match con esito:", len(labels), "| P(vince p1):", round(sum(labels)/len(labels), 3))
print("distinte -> specie:", len(sp), "mosse:", len(mv), "abilità:", len(ab), "strumenti:", len(it), "nature:", len(na))

import json, collections
nat = collections.Counter()
for line in open("data/matches.jsonl", encoding="utf-8"):
    for t in ("team1","team2"):
        for mon in json.loads(line)[t]:
            nat[mon["nature"]] += 1
for k, n in nat.most_common():
    print(n, repr(k))