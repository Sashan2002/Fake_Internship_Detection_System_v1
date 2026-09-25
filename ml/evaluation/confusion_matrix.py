"""
Confusion-matrix visualisation helper, kept separate from metrics.py so
plotting (which needs matplotlib) can be skipped in headless/CI evaluation
runs that only need the numeric metrics.
"""
import numpy as np


def plot_confusion_matrix(cm, class_names=("legitimate", "fraudulent"), output_path=None):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    return fig
