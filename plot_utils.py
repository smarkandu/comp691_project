import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def save_histogram(data, bins, title, xlabel, output_path):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_label_sample_grid(
    df,
    label_matrix,
    label_cols,
    name_col,
    image_root,
    output_dir,
    samples_per_label=4,
    random_seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    num_labels = len(label_cols)

    fig, axes = plt.subplots(
        num_labels,
        samples_per_label + 1,
        figsize=(16, 3 * num_labels),
    )

    if num_labels == 1:
        axes = np.array([axes])

    for row, (idx, label) in enumerate(enumerate(label_cols, start=1)):

        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.5,
            0.5,
            f"{idx}. {label}",
            ha="center",
            va="center",
            fontsize=12,
        )

        positive_rows = df[label_matrix[label] == 1]

        if len(positive_rows) == 0:
            for col in range(1, samples_per_label + 1):
                axes[row, col].axis("off")
                axes[row, col].text(0.5, 0.5, "No sample", ha="center", va="center")
            continue

        sampled_rows = positive_rows.sample(
            n=min(samples_per_label, len(positive_rows)),
            random_state=random_seed,
        )

        for i in range(samples_per_label):
            ax = axes[row, i + 1]
            ax.axis("off")

            if i < len(sampled_rows):
                img_name = sampled_rows.iloc[i][name_col]
                img_path = os.path.join(image_root, str(img_name))

                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(os.path.basename(str(img_name)), fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
                    ax.set_title(str(img_name), fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    plt.tight_layout()
    out_file = os.path.join(output_dir, "all_labels_grid.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved visualization to: {out_file}")

import os

def save_training_plots(history, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    save_curve(
        history["train_batch_loss"],
        "Per-batch Training Loss",
        "Batch",
        "Loss",
        os.path.join(out_dir, "train_batch_loss.png"),
    )

    save_curve(
        history["train_batch_map"],
        "Per-batch Training mAP",
        "Batch",
        "mAP",
        os.path.join(out_dir, "train_batch_map.png"),
    )

    save_curve(
        history["val_epoch_loss"],
        "Per-epoch Validation Loss",
        "Epoch",
        "Loss",
        os.path.join(out_dir, "val_epoch_loss.png"),
    )

    save_curve(
        history["val_epoch_map"],
        "Per-epoch Validation mAP",
        "Epoch",
        "mAP",
        os.path.join(out_dir, "val_epoch_map.png"),
    )

# Plot saving
def save_curve(values, title, xlabel, ylabel, out_path, x=None, marker=None):
    plt.figure()
    
    if x is None:
        plt.plot(values, marker=marker)
    else:
        plt.plot(x, values, marker=marker)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

