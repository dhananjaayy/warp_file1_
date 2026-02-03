import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
df = df.drop(columns=['Device_ID'])

x = df[['device_age_months','battery_capacity_mah']].values
y = df['thermal_stress_index'].values

y = (y > y.mean()).astype(int)

x_train, x1, y_train, y1 = train_test_split(
    x,y, test_size=0.30, random_state=42
)

x2, x_test, y2, y_test = train_test_split(
    x1, y1, test_size=0.50, random_state=42
)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

x2 = torch.tensor(x2, dtype=torch.float32)
y2 = torch.tensor(y2, dtype=torch.float32).view(-1,1)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,5),
            nn.ReLU(),
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,15),
            nn.ReLU(),
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        return self.model(x)

model = BinaryClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss  = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if(epoch+1) % 50 == 0:
        print(f"Epoch{epoch+1},loss: {loss.item():.4f}")


def evaluate(model, X, y, threshold=0.5):
    with torch.no_grad():
        probs = model(X).numpy()
        preds = (probs >= threshold).astype(int)

    y_true = y.numpy()
    cm = confusion_matrix(y_true, preds)
    precision = precision_score(y_true, preds,zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds,zero_division=0)

    return cm, precision, recall, f1


for name, xd, yd in [("Train", x_train, y_train),
                     ("Validation", x2, y2),
                     ("Test", x_test, y_test)]:
    cm, p, r, f = evaluate(model, xd, yd)
    print(f"\n{name} Set")
    print("Confusion Matrix:\n", cm)
    print(f"Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")


thresholds = np.arange(0, 1.05, 0.05)
precisions, recalls = [], []

for t in thresholds:
    _, p, r, _ = evaluate(model, x2, y2, threshold=t)
    precisions.append(p)
    recalls.append(r)

plt.plot(recalls, precisions, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid()
plt.show()



x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
y_min, y_max = x[:,1].min()-1, x[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = torch.tensor(
    np.c_[xx.ravel(), yy.ravel()],
    dtype=torch.float32
)

with torch.no_grad():
    zz = model(grid).numpy().reshape(xx.shape)

plt.contourf(xx, yy, zz, levels=50, cmap='coolwarm', alpha=0.6)
plt.scatter(x[:,0], x[:,1], c=y, edgecolors='k', cmap='coolwarm')
plt.title("Decision Boundary")
plt.savefig("file.png")
plt.close()
