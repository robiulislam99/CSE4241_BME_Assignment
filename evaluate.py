# evaluate.py
# Usage: python evaluate.py --model results/best_model.h5 --data data.npz --outdir results/eval
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def collapse_preds(y_pred):
    # y_pred: (N, L, 3) -> flatten per residue
    preds = np.argmax(y_pred, axis=-1).flatten()
    return preds

def collapse_true(y_true):
    trues = np.argmax(y_true, axis=-1).flatten()
    return trues

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data.npz")
    p.add_argument("--outdir", default="results/eval")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = np.load(args.data, allow_pickle=True)
    X_test, y_test = d['X_test'], d['y_test']

    model = load_model(args.model)
    y_pred = model.predict(X_test)

    y_pred_flat = collapse_preds(y_pred)
    y_true_flat = collapse_true(y_test)

    # Remove padded positions: assume padding token 0 in X_test
    mask = (X_test.flatten() != 0)
    y_pred_masked = y_pred_flat[mask]
    y_true_masked = y_true_flat[mask]

    labels = ["H", "E", "C"]
    print("Classification report:")
    print(classification_report(y_true_masked, y_pred_masked, target_names=labels, digits=4))

    cm = confusion_matrix(y_true_masked, y_pred_masked)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (per residue)")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png"))
    plt.close()

    print("Saved confusion matrix to", args.outdir)
