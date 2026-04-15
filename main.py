import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ======================================================
# 1. FLEXIBLE CNN ENCODER
# ======================================================
class FlexibleCNNEncoder(nn.Module):
    def __init__(self, latent_dim=128, c1=32, c2=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# ======================================================
# 2. CLASSIFIER WRAPPER
# ======================================================
class EncoderClassifier(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


# ======================================================
# 3. MNIST DATA
# ======================================================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# ======================================================
# 4. DEVICE
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# 5. TRAIN FUNCTION
# ======================================================
def train_model(model, epochs=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits, z = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        print(f"{model.encoder.__class__.__name__} | "
              f"Epoch {epoch} | Loss {total_loss:.4f} | Acc {correct/total:.4f}")

    return model


# ======================================================
# 6. MODEL DEFINITIONS
# ======================================================

latent_dim = 128

# 🔵 Model 1: CNN baseline
model1 = EncoderClassifier(
    FlexibleCNNEncoder(latent_dim=latent_dim, c1=32, c2=64)
)

# 🔵 Model 2: SAME architecture, different init
model2 = EncoderClassifier(
    FlexibleCNNEncoder(latent_dim=latent_dim, c1=32, c2=64)
)

# 🟣 Model 3: DIFFERENT capacity CNN
model3 = EncoderClassifier(
    FlexibleCNNEncoder(latent_dim=latent_dim, c1=16, c2=128)
)


# ======================================================
# 7. TRAIN ALL MODELS
# ======================================================
print("\nTraining Model 1")
model1 = train_model(model1)

print("\nTraining Model 2")
model2 = train_model(model2)

print("\nTraining Model 3")
model3 = train_model(model3)


# ======================================================
# 8. EMBEDDING EXTRACTION
# ======================================================
def extract_embeddings(model):
    model.eval()
    model = model.to(device)

    Z, Y = [], []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            _, z = model(x)

            Z.append(z.cpu().numpy())
            Y.append(y.numpy())

    return np.concatenate(Z), np.concatenate(Y)


X1, y1 = extract_embeddings(model1)
X2, y2 = extract_embeddings(model2)
X3, y3 = extract_embeddings(model3)


# ======================================================
# 9. SAVE EMBEDDINGS
# ======================================================
np.save("./data/mnist_emb_model1.npy", X1)
np.save("./data/mnist_emb_model2.npy", X2)
np.save("./data/mnist_emb_model3.npy", X3)
np.save("./data/mnist_labels.npy", y1)

print("\nSaved embeddings:")
print("Model1:", X1.shape)
print("Model2:", X2.shape)
print("Model3:", X3.shape)