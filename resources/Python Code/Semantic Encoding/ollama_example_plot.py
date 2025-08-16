import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
csv_path = os.path.join(csv_dir, 'ollama_example.csv')
plot_dir = os.path.join(os.path.dirname(__file__), 'plot')
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, 'ollama_example.png')

sentences = []
vectors = []
with open(csv_path, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        sentences.append(row[0])
        vectors.append([float(x) for x in row[1:]])
vectors = np.array(vectors)

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, (vec, sent) in enumerate(zip(vectors_2d, sentences)):
    plt.arrow(0, 0, vec[0], vec[1], head_width=0.08, head_length=0.12, fc='C0', ec='C0', length_includes_head=True)
    plt.text(vec[0], vec[1], sent, fontsize=9)
plt.title('ollama_example Embeddings (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
