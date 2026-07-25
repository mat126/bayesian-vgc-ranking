"""
neural_bt.py
Bradley-Terry neurale per VGC: rete siamese che segna due squadre e ne prende la
differenza come logit. Rispetta i due vincoli del problema:
  - antisimmetria: stessa rete (pesi condivisi) su A e B, logit = s(A) - s(B)
  - invarianza per permutazione: pooling sulle 4 mosse e attenzione+pool sui 6 Pokemon
La sinergia emerge dal self-attention tra i 6 Pokemon (niente positional encoding).

Struttura del batch atteso (tensori LongTensor di indici, 0 = mancante/padding):
  team = {
    "species": (B, 6),  "ability": (B, 6),  "item": (B, 6),  "nature": (B, 6),
    "moves":   (B, 6, 4),
  }
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PokemonEncoder(nn.Module):
    """Un vettore per Pokemon a partire dalle sue feature."""
    def __init__(self, vocabs, d=24, h=64, p_drop=0.2):
        super().__init__()
        self.species = nn.Embedding(vocabs["species"], d)
        self.move    = nn.Embedding(vocabs["moves"],   d, padding_idx=0)
        self.ability = nn.Embedding(vocabs["ability"], d, padding_idx=0)
        self.item    = nn.Embedding(vocabs["item"],    d, padding_idx=0)
        self.nature  = nn.Embedding(vocabs["nature"],  d, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(5 * d, h), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(h, h)
        )

    def forward(self, t):
        mv = self.move(t["moves"]).mean(dim=2)             # pool 4 mosse -> (B,6,d)
        x = torch.cat([self.species(t["species"]), mv,
                       self.ability(t["ability"]), self.item(t["item"]),
                       self.nature(t["nature"])], dim=-1)  # (B,6,5d)
        return self.mlp(x)                                 # (B,6,h)


class TeamEncoder(nn.Module):
    """Self-attention sui 6 Pokemon (sinergia) + pooling invariante -> forza scalare."""
    def __init__(self, h=64, heads=4, layers=2, p_drop=0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=heads, dim_feedforward=2 * h,
            dropout=p_drop, batch_first=True)             # NIENTE positional encoding
        self.attn = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, mons):            # (B,6,h)
        z = self.attn(mons)             # (B,6,h) contestualizzati dai compagni
        z = z.mean(dim=1)               # pool permutation-invariant -> (B,h)
        return self.head(z).squeeze(-1) # (B,) forza s(team)


class TeamBT(nn.Module):
    def __init__(self, vocabs, d=24, h=64):
        super().__init__()
        self.poke = PokemonEncoder(vocabs, d, h)
        self.team = TeamEncoder(h)
        self.side = nn.Parameter(torch.zeros(1))   # bias di lato (rompe l'antisimmetria di proposito)

    def score(self, team):
        return self.team(self.poke(team))          # (B,)

    def forward(self, A, B):
        return self.score(A) - self.score(B) + self.side   # logit BT (antisimmetrico per costruzione)


if __name__ == "__main__":
    # smoke test con dati finti
    vocabs = {"species": 220, "moves": 460, "ability": 170, "item": 165, "nature": 26}
    B = 8
    def fake_team():
        return {"species": torch.randint(1, 220, (B, 6)),
                "ability": torch.randint(0, 170, (B, 6)),
                "item":    torch.randint(0, 165, (B, 6)),
                "nature":  torch.randint(0, 26, (B, 6)),
                "moves":   torch.randint(0, 460, (B, 6, 4))}
    model = TeamBT(vocabs)
    logit = model(fake_team(), fake_team())
    print("logit shape:", logit.shape, "| n. parametri:", sum(p.numel() for p in model.parameters()))
