# utils_generate_synthetic.py
# Creates a tiny synthetic sequences.fasta and labels.txt for testing the pipeline.
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def random_seq(length):
    return "".join(random.choices(AMINO_ACIDS, k=length))

def synthetic_label(length):
    # Very naive: first third helix H, second third sheet E, last third coil C (for testing only)
    third = max(1, length // 3)
    return "H"*third + "E"*third + "C"*(length - 2*third)

def generate(n=30, out_fasta="sequences.fasta", out_labels="labels.txt"):
    records = []
    labels = []
    for i in range(n):
        L = random.randint(50, 200)
        s = random_seq(L)
        lbl = synthetic_label(L)
        records.append(SeqRecord(Seq(s), id=f"synt_{i}", description="synthetic"))
        labels.append(lbl)

    SeqIO.write(records, out_fasta, "fasta")
    with open(out_labels, "w") as f:
        for l in labels:
            f.write(l + "\n")
    print(f"Generated {n} sequences to {out_fasta} and labels to {out_labels}")

if __name__ == "__main__":
    generate()
