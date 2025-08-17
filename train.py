# train.py
# Usage: python train.py --data data.npz --epochs 20 --outdir results
import numpy as np
import os
import argparse
from model import get_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import imageio
from tqdm import tqdm

class WeightSnapshotCallback(Callback):
    def __init__(self, out_dir="results/weights_snapshots"):
        super().__init__()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Take snapshots of first layer weights (embedding or conv)
        weights = None
        # prefer conv1d or embedding
        for layer in self.model.layers:
            if layer.name.startswith("conv1d") or layer.__class__.__name__ == "Embedding":
                w = layer.get_weights()[0]
                weights = w
                break
        if weights is not None:
            # For visualization, reduce to 2D image by normalizing
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,4))
            ax.imshow(weights.T, aspect='auto')
            ax.set_title(f"weights_epoch_{epoch+1}")
            ax.set_xlabel("input dim")
            ax.set_ylabel("output dim")
            fpath = os.path.join(self.out_dir, f"weights_epoch_{epoch+1}.png")
            plt.tight_layout()
            fig.savefig(fpath)
            plt.close(fig)
        self.epoch += 1

def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Loss plot (per-residue categorical crossentropy)
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # Accuracy plot (we compute per-residue accuracy via metrics if available)
    if 'accuracy' in history.history or 'val_accuracy' in history.history:
        plt.figure()
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='train_acc')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend(); plt.title("Accuracy")
        plt.savefig(os.path.join(out_dir, "acc_curve.png"))
        plt.close()

def make_gif(images_dir, out_gif):
    pngs = sorted([os.path.join(images_dir, p) for p in os.listdir(images_dir) if p.endswith(".png")])
    if not pngs:
        print("No weight snapshot images found for GIF.")
        return
    imgs = [imageio.imread(p) for p in pngs]
    imageio.mimsave(out_gif, imgs, fps=1)
    print("Saved GIF:", out_gif)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X_train, y_train = d['X_train'], d['y_train']
    X_val, y_val = d['X_val'], d['y_val']

    maxlen = X_train.shape[1]
    model = get_model(maxlen=maxlen)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    os.makedirs(args.outdir, exist_ok=True)
    ckpt = ModelCheckpoint(os.path.join(args.outdir, "best_model.h5"), save_best_only=True, monitor="val_loss")
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    snap = WeightSnapshotCallback(out_dir=os.path.join(args.outdir, "weights_snapshots"))

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=args.epochs,
                        batch_size=args.batch,
                        callbacks=[ckpt, early, snap])

    # save history
    np.save(os.path.join(args.outdir, "history.npy"), history.history)

    # plots
    plot_history(history, args.outdir)

    # GIF
    make_gif(os.path.join(args.outdir, "weights_snapshots"), os.path.join(args.outdir, "weights_evolution.gif"))
