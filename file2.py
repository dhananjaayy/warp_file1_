import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("SAR Rental.csv")
df.head()
df.info()
df.shape

df = df.select_dtypes(include=[np.number])

df = df.fillna(df.mean(numeric_only=True))

X = df.iloc[:, :-1].values   
y = df.iloc[:, -1].values   

print("x shape:", X.shape)
print("y shape:", y.shape)


print(X.shape)
print(y.shape)

X = (X - X.mean(axis=0)) / X.std(axis=0)

w = np.random.randn(X.shape[1])
b = 0
lr = 0.01
epochs = 1000
n = len(X)
if n ==0:
    raise ValueError("Datashet is empty after preprocessing")

for _ in range(epochs):
    y_pred = np.dot(X, w) + b

    dw = (-2/n) * np.dot(X.T, (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    w -= lr * dw
    b -= lr * db

y_pred = np.dot(X, w) + b

def predict(X):
    return np.dot(X, w) + b
      
