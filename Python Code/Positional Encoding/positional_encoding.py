import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    """Generate positional encoding matrix for a sequence length and model dimension."""
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

if __name__ == "__main__":
    import os
    import csv
    seq_len = 50
    d_model = 16
    pe = positional_encoding(seq_len, d_model)
    # Save positional encoding to CSV (matrix format, with total sum column)
    os.makedirs("csv", exist_ok=True)
    csv_path = "csv/positional_encoding.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"dim_{i}" for i in range(d_model)] + ["mean"]
        writer.writerow(["position"] + header)
        for pos in range(seq_len):
            row = list(pe[pos])
            mean = sum(row) / d_model
            writer.writerow([pos] + row + [mean])
    print(f"Positional encoding data saved as {csv_path}")

    # Save positional encoding to CSV (sequence format)
    csv_seq_path = "csv/positional_encoding_sequence.csv"
    with open(csv_seq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "dimension", "value"])
        for pos in range(seq_len):
            for dim in range(d_model):
                writer.writerow([pos, dim, pe[pos, dim]])
    print(f"Positional encoding sequence data saved as {csv_seq_path}")

