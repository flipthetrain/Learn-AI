import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Paths
csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
csv_path = os.path.join(csv_dir, 'positional_encoding.csv')
plot_dir = os.path.join(os.path.dirname(__file__), 'plot')
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, 'positional_encoding.png')

# Read CSV
positions = []
values = []
with open(csv_path, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        positions.append(int(row[0]))
        values.append([float(x) for x in row[1:]])
values = np.array(values)



# Remove redundant dim 17 (index 16) if present
d_model = values.shape[1]
redundant_dim = 16 if d_model >= 17 else None
plot_dims = [i for i in range(d_model) if i != redundant_dim]

fig, axes = plt.subplots(len(plot_dims) + 2, 1, figsize=(10, 2 * (len(plot_dims) + 2)), sharex=True)
for ax_idx, i in enumerate(plot_dims):
    axes[ax_idx].plot(positions, values[:, i], label=f"Dim {i+1}")
    axes[ax_idx].set_ylabel(f"Dim {i+1}")
    axes[ax_idx].legend(loc='upper right', fontsize='small')
    # Add dots and position labels
    for pos, val in zip(positions, values[:, i]):
        axes[ax_idx].scatter(pos, val, color='red', s=20, zorder=3)
        axes[ax_idx].text(pos, val, str(pos), fontsize=7, color='black', ha='center', va='bottom')

# Mean plot (mean across all dimensions)
mean_ax = axes[len(plot_dims)]
mean_vals = values.mean(axis=1)
mean_ax.plot(positions, mean_vals, label="Mean(all dims)", color='green')
mean_ax.set_ylabel("Mean")
mean_ax.legend(loc='upper right', fontsize='small')
# Add dots and position labels
for pos, val in zip(positions, mean_vals):
    mean_ax.scatter(pos, val, color='red', s=20, zorder=3)
    mean_ax.text(pos, val, str(pos), fontsize=7, color='black', ha='center', va='bottom')

# Overlay plot (all non-redundant dimensions)
overlay_ax = axes[-1]
for i in plot_dims:
    overlay_ax.plot(positions, values[:, i], label=f"Dim {i+1}")
overlay_ax.set_ylabel("Overlay")
overlay_ax.set_xlabel("Position")

# Move legend further down (about 1 inch below the plot)
fig_height_inches = fig.get_size_inches()[1]
# Move legend down by about 3 inches (or more if needed)
legend_offset = -0.35 - (3.0 / fig_height_inches)  # Move down by about 3 inches
overlay_ax.legend(loc='lower center', bbox_to_anchor=(0.5, legend_offset), fontsize='small', ncol=4, frameon=False)

fig.suptitle("Positional Encoding: Stacked, Sum, and Overlay Plots")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(plot_path)
plt.close(fig)
print(f"Plot saved to {plot_path}")
