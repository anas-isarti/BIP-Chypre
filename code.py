#BIP CHYPRE
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load CSV (after uploading to /content/)
df = pd.read_csv('/content/spam_ham_dataset.csv')

# 2. Simple tokenization
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

df['tokens'] = df['text'].apply(tokenize)
df['processed_text'] = df['tokens'].apply(lambda toks: ' '.join(toks))

# 3. Bag-of-Words vectorization with limited vocabulary
vectorizer = CountVectorizer(max_features=10_000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label_num'].values  # 0 = ham, 1 = spam

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. PyTorch Dataset & DataLoader
class SMSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SMSDataset(X_train, y_train)
test_ds  = SMSDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

# 6. Logistic Regression model with optional dropout
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, p_drop=0.2):
        super().__init__()
        self.drop   = nn.Dropout(p_drop)
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        x = self.drop(x)
        return torch.sigmoid(self.linear(x)).squeeze(1)

model = LogisticRegressionModel(X_train.shape[1], p_drop=0.2)

# 7. Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt  = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

criterion = FocalLoss(alpha=1.0, gamma=2.0)

# 8. Optimizer with L2 weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 9. Training & evaluation functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            predicted = (preds >= 0.5).float()
            correct  += (predicted == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# 10. Training loop with early stopping and real-time plots
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
n_epochs = 100
patience, best_val_loss, streak = 5, float('inf'), 0

train_losses, val_losses, val_accs = [], [], []

for epoch in range(1, n_epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        streak = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        streak += 1
        if streak >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Real-time plotting
    clear_output(wait=True)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss over Epochs'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Epoch {epoch} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# 11. Reload best model and final evaluation
model.load_state_dict(torch.load('best_model.pt'))
y_pred = (model(torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)) >= 0.5).int().cpu()

print("\n=== Final Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print(f"=== Overall Accuracy: {accuracy_score(y_test, y_pred):.4f} ===")

# 12. Plot confusion matrix with heatmap
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# 13. Print token lists for inspection
print("\n=== Token lists (first 10 examples) ===")
for toks in df['tokens'].head(10):
    print(toks)
