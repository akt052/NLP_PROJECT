import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from preprocess import load_data, tokenize
from dataset import TranslationDataset, collate_fn
from utils import build_vocab
from models.lstm import Encoder, Decoder, Seq2Seq
from evaluate import compute_metrics


device = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output = output.reshape(-1, output.shape[-1])
        trg_y = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def evaluate_model(model, loader, trg_vocab):
    model.eval()
    preds, refs = [], []

    inv_vocab = {v: k for k, v in trg_vocab.items()}

    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)

            for i in range(src.size(0)):
                h, c = model.enc(src[i:i+1])

                trg_idx = [trg_vocab["<sos>"]]

                for _ in range(50):
                    inp = torch.tensor([[trg_idx[-1]]]).to(device)
                    out, h, c = model.dec(inp, h, c)
                    pred = out.argmax(2).item()
                    trg_idx.append(pred)

                    if pred == trg_vocab["<eos>"]:
                        break

                pred_sentence = " ".join(
                    [inv_vocab.get(tok, "<unk>") for tok in trg_idx[1:-1]]
                )

                ref_sentence = " ".join(
                    [inv_vocab.get(tok.item(), "<unk>") for tok in trg[i][1:-1]]
                )

                preds.append(pred_sentence)
                refs.append(ref_sentence)

    return compute_metrics(preds, refs)


# ---- K-FOLD ----

data = load_data("data/train.csv")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    print(f"\n--- Fold {fold+1} ---")

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    # vocab (IMPORTANT: no tokenization here)
    src_vocab = build_vocab(
        [{"src": tokenize(d["src"])} for d in train_data], "src"
    )
    trg_vocab = build_vocab(
        [{"trg": tokenize(d["trg"])} for d in train_data], "trg"
    )

    train_ds = TranslationDataset(train_data, src_vocab, trg_vocab, tokenize)
    val_ds = TranslationDataset(val_data, src_vocab, trg_vocab, tokenize)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b, src_vocab["<pad>"], trg_vocab["<pad>"]
        )
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b, src_vocab["<pad>"], trg_vocab["<pad>"]
        )
    )

    enc = Encoder(len(src_vocab), 256, 512)
    dec = Decoder(len(trg_vocab), 256, 512)
    model = Seq2Seq(enc, dec).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab["<pad>"])

    for epoch in range(25):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Loss = {loss:.3f}")

    bleu, chrf, score = evaluate_model(model, val_loader, trg_vocab)
    print(f"BLEU: {bleu:.3f}, chrF: {chrf:.3f}, score: {score:.3f}")

    scores.append(score)

print(f"\nFinal CV Score: {sum(scores)/len(scores):.4f}")