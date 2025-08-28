
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# Ensure output folders exist
os.makedirs('csv', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Read weights and biases from CSV
weights = []
with open('csv/xor_nn_weights.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        weights.append(row)

# Extract weights and biases
W1 = np.zeros((2, 2))
b1 = np.zeros((1, 2))
W2 = np.zeros((2, 1))
b2 = np.zeros((1, 1))
for row in weights:
    if row['Layer'] == 'Input-Hidden':
        i = int(row['From'][1]) - 1
        j = int(row['To'][1]) - 1
        W1[i, j] = float(row['Weight'])
    elif row['Layer'] == 'Hidden-Output':
        j = int(row['From'][1]) - 1
        W2[j, 0] = float(row['Weight'])
    elif row['Layer'] == 'Bias' and row['To'].startswith('h'):
        j = int(row['To'][1]) - 1
        b1[0, j] = float(row['Weight'])
    elif row['Layer'] == 'Bias' and row['To'] == 'y':
        b2[0, 0] = float(row['Weight'])

# Read XOR data
data = np.loadtxt('csv/xor_data.csv', delimiter=',', skiprows=1)
X_data = data[:, :2]
y_data = data[:, 2]

# Define the forward pass for plotting decision boundary
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a2

def predict(X):
    return (forward(X) > 0.5).astype(int)

# Plot 1: Data and decision boundary
plt.figure(figsize=(6, 6))
plt.scatter(X_data[y_data == 0][:, 0], X_data[y_data == 0][:, 1], c='red', label='Class 0', alpha=0.6)
plt.scatter(X_data[y_data == 1][:, 0], X_data[y_data == 1][:, 1], c='blue', label='Class 1', alpha=0.6)
plt.scatter(X_data[:, 0], X_data[:, 1], c='black', s=20, marker='o', label='Data Points', edgecolors='white', linewidths=0.5)
xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, zz, alpha=0.2, levels=[-0.1, 0.5, 1.1], colors=['red', 'blue'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Data and Neural Network Decision Boundary')
plt.legend()
plt.tight_layout()
plt.savefig('plots/xor_nn_decision_boundary.png')
print('Plot saved as plots/xor_nn_decision_boundary.png')

# Plot 2: Neural network architecture with weights
def plot_network(W1, b1, W2, b2, filename='plots/xor_nn_architecture.png'):
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')
    input_nodes = [(0, 0.7), (0, 0.3)]
    hidden_nodes = [(0.5, 0.8), (0.5, 0.2)]
    output_node = (1, 0.5)
    for i, pos in enumerate(input_nodes):
        ax.add_patch(plt.Circle(pos, 0.05, color='lightgray', ec='black', zorder=2))
        ax.text(pos[0]-0.08, pos[1], f'x{i+1}', fontsize=12, va='center')
    for i, pos in enumerate(hidden_nodes):
        ax.add_patch(plt.Circle(pos, 0.05, color='lightblue', ec='black', zorder=2))
        ax.text(pos[0], pos[1]+0.09, f'h{i+1}', fontsize=12, ha='center')
    ax.add_patch(plt.Circle(output_node, 0.05, color='lightgreen', ec='black', zorder=2))
    ax.text(output_node[0]+0.08, output_node[1], 'y', fontsize=12, va='center')
    for i, inp in enumerate(input_nodes):
        for j, hid in enumerate(hidden_nodes):
            w = W1[i, j]
            ax.annotate('', xy=hid, xytext=inp, arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
            ax.text((inp[0]+hid[0])/2-0.03, (inp[1]+hid[1])/2, f'{w:.2f}', fontsize=10, color='purple')
    for j, hid in enumerate(hidden_nodes):
        w = W2[j, 0]
        ax.annotate('', xy=output_node, xytext=hid, arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        ax.text((hid[0]+output_node[0])/2+0.03, (hid[1]+output_node[1])/2, f'{w:.2f}', fontsize=10, color='purple')
    for j, hid in enumerate(hidden_nodes):
        ax.text(hid[0]-0.05, hid[1]-0.09, f'b={b1[0, j]:.2f}', fontsize=9, color='brown')
    ax.text(output_node[0]+0.05, output_node[1]-0.09, f'b={b2[0, 0]:.2f}', fontsize=9, color='brown')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 1)
    plt.title('Simple XOR Neural Network Architecture with Weights')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

plot_network(W1, b1, W2, b2)
print('Network architecture plot saved as plots/xor_nn_architecture.png')
