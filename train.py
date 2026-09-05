import torch
import torch.nn.functional as F
from models import EvoPepHybridGenerator, AA_TO_IDX

SEED_SCAFFOLDS = [
    "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
    "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG",
    "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPSKKKKK",
    "HAEGTFTSDVSISYLEGQAAKEFIAWLVKGR",
]

VOCAB_SIZE = 20
EPOCHS = 200
LR = 1e-3


def encode(seq):
    return [AA_TO_IDX.get(aa, 0) for aa in seq]


def main():
    torch.manual_seed(0)
    model = EvoPepHybridGenerator(vocab_size=VOCAB_SIZE, d_model=64, nhead=4, mpnn_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    encoded = [encode(s) for s in SEED_SCAFFOLDS]
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for seq in encoded:
            x = torch.tensor([seq[:-1]], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch}/{EPOCHS} avg_loss {total_loss / len(encoded):.4f}")
    torch.save(model.state_dict(), "pretrained_hybrid.pth")
    print("saved pretrained_hybrid.pth")


if __name__ == "__main__":
    main()
