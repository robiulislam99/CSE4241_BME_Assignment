# preprocess.py
# Usage: python preprocess.py --fasta sequences.fasta --labels labels.txt --out data.npz

import numpy as np
from Bio import SeqIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import os

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard
aa_to_idx = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # reserve 0 for padding

label_map = {"H":0, "E":1, "C":2}  # helix, sheet, coil

def read_fasta(fname):
    records = list(SeqIO.parse(fname, "fasta"))
    seqs = [str(r.seq) for r in records]
    ids  = [r.id for r in records]
    return ids, seqs

def read_labels(fname):
    with open(fname) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def encode_sequence(seq):
    # map unknown AA to 0 (padding is 0)
    return [aa_to_idx.get(ch, 0) for ch in seq]

def encode_labels(labels):
    return [label_map.get(ch, 2) for ch in labels]  # default coil if unknown

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="data.npz")
    parser.add_argument("--maxlen", type=int, default=700)  # adjust as needed
    args = parser.parse_args()

    ids, seqs = read_fasta(args.fasta)
    labels_lines = read_labels(args.labels)

    assert len(seqs) == len(labels_lines), "Number of sequences and label lines must match."

    X = [encode_sequence(s) for s in seqs]
    y = [encode_labels(l) for l in labels_lines]

    Xp = pad_sequences(X, maxlen=args.maxlen, padding="post", truncating="post", value=0)
    # For per-residue classification, we will pad labels with 2 (coil) by default
    y_padded = pad_sequences(y, maxlen=args.maxlen, padding="post", truncating="post", value=2)

    # Convert y to one-hot per residue: shape (N, maxlen, 3)
    from tensorflow.keras.utils import to_categorical
    y_onehot = to_categorical(y_padded, num_classes=3)

    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(Xp, y_onehot, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    np.savez_compressed(args.out,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test,
                        ids=ids)

    print(f"Saved processed data to {args.out}")
    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
