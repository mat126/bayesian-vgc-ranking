"""
neural_data.py
Prepara il dataset per la rete neurale (neural_bt.TeamBT).

Trasforma matches_clean.jsonl nei tensori di indici attesi dal modello:
  team = { "species":(B,6), "ability":(B,6), "item":(B,6), "nature":(B,6), "moves":(B,6,4) }
Convenzioni:
  - indice 0 = PADDING / MANCANTE / raro (UNK). I vocabolari partono da 1.
  - le mosse vengono impaccate/troncate a 4, i Pokemon a 6.
  - i vocabolari si costruiscono SOLO sul train (niente leakage); val/test mappano
    i simboli sconosciuti a 0.
  - soglia di frequenza opzionale: i livelli rari confluiscono su 0 (token "raro").

Uso tipico (dal training):
  from neural_data import make_loaders, vocab_sizes
  train_dl, val_dl, vocab = make_loaders("data/matches_clean.jsonl", batch_size=256)
  model = TeamBT(vocab_sizes(vocab))
"""
from __future__ import annotations
import collections, json, random

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:                      # permette di costruire/testare i vocabolari senza torch
    _HAS_TORCH = False
    Dataset = object

FIELDS = ("species", "ability", "item", "nature", "moves")
SINGLE = ("species", "ability", "item", "nature")   # un valore per Pokemon
DEFAULT_MIN_FREQ = {"species": 1, "ability": 5, "item": 5, "nature": 1, "moves": 5}


def load_matches(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("label") is not None:      # scarta i pareggi
                out.append(m)
    return out


def build_vocab(matches, min_freq=None):
    """Vocabolari {livello -> indice>=1} per ciascun campo, con 0 riservato a PAD/UNK."""
    mf = {**DEFAULT_MIN_FREQ, **(min_freq or {})}
    cnt = {f: collections.Counter() for f in FIELDS}
    for m in matches:
        for tk in ("team1", "team2"):
            for mon in m[tk]:
                for f in SINGLE:
                    v = mon.get(f)
                    if v:
                        cnt[f][v] += 1
                for mv in mon.get("moves", []):
                    if mv:
                        cnt["moves"][mv] += 1
    vocab = {}
    for f in FIELDS:
        levels = sorted(lv for lv, n in cnt[f].items() if n >= mf.get(f, 1))
        vocab[f] = {lv: i + 1 for i, lv in enumerate(levels)}    # 0 riservato
    return vocab


def vocab_sizes(vocab):
    """Numero di righe di embedding per campo (= max indice + 1, cioe' len + 1 per il PAD)."""
    return {f: len(vocab[f]) + 1 for f in FIELDS}


def save_vocab(vocab, path):
    json.dump(vocab, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def load_vocab(path):
    return json.load(open(path, encoding="utf-8"))


def encode_team(team, vocab):
    """Squadra -> dict di liste di indici (species/ability/item/nature: 6; moves: 6x4)."""
    out = {f: [] for f in FIELDS}
    for mon in team[:6]:
        for f in SINGLE:
            out[f].append(vocab[f].get(mon.get(f), 0))
        mvs = [vocab["moves"].get(x, 0) for x in mon.get("moves", [])][:4]
        mvs += [0] * (4 - len(mvs))            # impacca a 4
        out["moves"].append(mvs)
    # impacca a 6 Pokemon (di norma gia' 6)
    while len(out["species"]) < 6:
        for f in SINGLE:
            out[f].append(0)
        out["moves"].append([0, 0, 0, 0])
    return out


class VGCDataset(Dataset):
    """Precalcola i tensori di indici per tutti i match. __getitem__ restituisce (A, B, y)."""
    def __init__(self, matches, vocab, augment=False):
        if not _HAS_TORCH:
            raise ImportError("PyTorch non disponibile: installa torch per usare VGCDataset.")
        self.augment = augment
        A = {f: [] for f in FIELDS}
        B = {f: [] for f in FIELDS}
        y = []
        for m in matches:
            a, b = encode_team(m["team1"], vocab), encode_team(m["team2"], vocab)
            for f in FIELDS:
                A[f].append(a[f]); B[f].append(b[f])
            y.append(m["label"])
        self.A = {f: torch.tensor(A[f], dtype=torch.long) for f in FIELDS}
        self.B = {f: torch.tensor(B[f], dtype=torch.long) for f in FIELDS}
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a = {f: self.A[f][i] for f in FIELDS}
        b = {f: self.B[f][i] for f in FIELDS}
        y = self.y[i]
        if self.augment and random.random() < 0.5:     # scambia A/B e inverti la label
            a, b, y = b, a, 1.0 - y
        return a, b, y


def split_matches(matches, val_frac=0.15, seed=0):
    """Split train/val deterministico (stesso seed -> stesso split in train ed eval)."""
    rng = random.Random(seed)
    idx = list(range(len(matches)))
    rng.shuffle(idx)
    cut = int((1 - val_frac) * len(idx))
    return [matches[i] for i in idx[:cut]], [matches[i] for i in idx[cut:]]


def make_loaders(path, batch_size=256, val_frac=0.15, min_freq=None,
                 augment=True, seed=0, num_workers=0):
    """Costruisce train/val DataLoader. Il vocabolario e' stimato SOLO sul train."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch non disponibile: installa torch.")
    matches = load_matches(path)
    train, val = split_matches(matches, val_frac, seed)

    vocab = build_vocab(train, min_freq)
    train_ds = VGCDataset(train, vocab, augment=augment)
    val_ds   = VGCDataset(val,   vocab, augment=False)

    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, generator=g)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)
    return train_dl, val_dl, vocab


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/matches_clean.jsonl"
    matches = load_matches(path)
    vocab = build_vocab(matches)
    print("match:", len(matches))
    print("dimensioni vocabolari (righe embedding, PAD incluso):", vocab_sizes(vocab))
    # anteprima encoding del primo team, senza torch
    enc = encode_team(matches[0]["team1"], vocab)
    print("esempio species:", enc["species"])
    print("esempio moves[0]:", enc["moves"][0])
    if _HAS_TORCH:
        train_dl, val_dl, vocab = make_loaders(path, batch_size=64)
        A, B, y = next(iter(train_dl))
        print("batch: species", tuple(A["species"].shape),
              "| moves", tuple(A["moves"].shape), "| y", tuple(y.shape))
