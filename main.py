import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import Lasso

# ======================================================
# 1. SUPERVISED ENCODER (TRAINED WITH LABELS)
# ======================================================


class EncoderClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1×28×28 → 32×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # → 32×14×14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),# → 64×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # → 64×7×7

            nn.Flatten(),                                         # → 64*7*7
            nn.Linear(64 * 7 * 7, latent_dim)                     # latent code z
        )

        # Classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, 1, 28, 28)
        """
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


# Load MNIST


transform = transforms.Compose([
    transforms.ToTensor(),                 # → (1, 28, 28)
    transforms.Lambda(lambda x: x.view(1, 28, 28))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Train supervised encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, z = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Acc: {correct/total:.4f}")

# ======================================================
# 2. EXTRACT EMBEDDINGS (LABEL-AWARE)
# ======================================================

model.eval()
embeddings = []
labels = []

with torch.no_grad():
    for x, y in train_loader:
        x = x.to(device)
        _, z = model(x)
        embeddings.append(z.cpu().numpy())
        labels.append(y.numpy())

X_emb = np.concatenate(embeddings, axis=0)
y_all = np.concatenate(labels, axis=0)

print("Embeddings shape:", X_emb.shape)

np.save("./data/" + "mnist_embeddings.npy", X_emb)
np.save("./data/" + "mnist_labels.npy", y_all)
print("Saved embeddings and labels to disk.")


