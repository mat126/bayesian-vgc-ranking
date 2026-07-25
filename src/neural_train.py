"""
neural_train.py
Addestra la rete TeamBT con early stopping sul log-loss di validazione.

  python neural_train.py --matches data/matches_clean.jsonl --epochs 60

Salva il miglior modello (per val log-loss) in neural_ckpt.pt, con dentro anche il
vocabolario e la configurazione, cosi' neural_eval.py ricostruisce tutto da solo.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn

from neural_data import make_loaders, vocab_sizes, save_vocab
from neural_bt import TeamBT


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    tot, correct, n = 0.0, 0, 0
    for A, B, y in dl:
        A, B, y = to_device(A, device), to_device(B, device), y.to(device)
        logit = model(A, B)
        tot += bce(logit, y).item()
        correct += ((logit > 0).float() == y).sum().item()
        n += y.numel()
    return tot / n, correct / n            # log-loss medio, accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="data/matches_clean.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)   # regolarizzazione (prior gaussiano sui pesi)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="neural_ckpt.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_dl, val_dl, vocab = make_loaders(a.matches, batch_size=a.batch_size,
                                           val_frac=a.val_frac, seed=a.seed, augment=True)
    sizes = vocab_sizes(vocab)
    print("dimensioni vocabolari:", sizes)

    model = TeamBT(sizes, d=a.d, h=a.h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for A, B, y in train_dl:
            A, B, y = to_device(A, device), to_device(B, device), y.to(device)
            opt.zero_grad()
            loss = bce(model(A, B), y)
            loss.backward()
            opt.step()
            run += loss.item(); nb += 1
        vloss, vacc = evaluate(model, val_dl, device)
        flag = ""
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        else:
            bad += 1
        print(f"ep {ep:02d}  train {run/nb:.4f}  val logloss {vloss:.4f}  val acc {vacc:.4f}{flag}")
        if bad >= a.patience:
            print(f"early stopping: nessun miglioramento da {a.patience} epoche")
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "vocab": vocab, "sizes": sizes,
                "config": {"d": a.d, "h": a.h},
                "val_frac": a.val_frac, "seed": a.seed, "best_val_logloss": best}, a.out)
    save_vocab(vocab, "neural_vocab.json")
    import numpy as np
    print(f"\nmiglior val log-loss: {best:.4f}  (baseline coin-flip {np.log(2):.4f})")
    print(f"checkpoint salvato in {a.out}")


if __name__ == "__main__":
    main()
