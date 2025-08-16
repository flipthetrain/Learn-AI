import numpy as np

# Generate random XOR data
np.random.seed(42)
X = np.random.randint(0, 2, (100, 2))
y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)

# Simple neural network for XOR
class SimpleXORNet:
    def __init__(self):
        # 2 input, 2 hidden, 1 output
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.zeros((1, 1))
        self.lr = 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        dz2 = output - y.reshape(-1, 1)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=10000):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if i % 1000 == 0:
                loss = np.mean((output - y.reshape(-1, 1)) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

if __name__ == "__main__":
    import os
    # Ensure output folder exists
    os.makedirs('csv', exist_ok=True)
    import matplotlib.pyplot as plt
    # Save sample data to file
    np.savetxt("csv/xor_data.csv", np.column_stack((X, y)), delimiter=",", header="x1,x2,label", comments="")
    # Load data from file
    data = np.loadtxt("csv/xor_data.csv", delimiter=",", skiprows=1)
    X_data = data[:, :2]
    y_data = data[:, 2]
    # Train network
    net = SimpleXORNet()
    net.train(X_data, y_data)
    # Test
    preds = net.predict(X_data)
    acc = np.mean(preds.flatten() == y_data)
    print(f"Accuracy: {acc * 100:.2f}%")
    # Show some predictions
    for i in range(5):
        print(f"Input: {X_data[i]}, Predicted: {preds[i][0]}, Actual: {int(y_data[i])}")

    # ...plotting code moved to xor_nn_plots.py...

    # Output weights and biases to CSV
    import csv
    # Collect all weights and biases into a list
    rows = []
    # Input to hidden
    for i in range(net.W1.shape[0]):
        for j in range(net.W1.shape[1]):
            rows.append(['Input-Hidden', f'x{i+1}', f'h{j+1}', f'{net.W1[i, j]:.6f}'])
    # Hidden to output
    for j in range(net.W2.shape[0]):
        rows.append(['Hidden-Output', f'h{j+1}', 'y', f'{net.W2[j, 0]:.6f}'])
    # Biases
    for j in range(net.b1.shape[1]):
        rows.append(['Bias', '1', f'h{j+1}', f'{net.b1[0, j]:.6f}'])
    rows.append(['Bias', '1', 'y', f'{net.b2[0, 0]:.6f}'])
    # Sort by 'To' then 'From'
    rows_sorted = sorted(rows, key=lambda r: (r[2], r[1]))
    with open('csv/xor_nn_weights.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer', 'From', 'To', 'Weight'])
        writer.writerows(rows_sorted)
    print('Weights and biases saved to csv/xor_nn_weights.csv')
