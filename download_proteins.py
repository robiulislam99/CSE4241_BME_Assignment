# download_proteins.py
# Usage: python download_proteins.py
# Make sure you installed biopython and replaced Entrez.email if necessary.

from Bio import Entrez
import sys

Entrez.email = "beingonair@gmail.com"  # <-- your email

def download_fasta(search_term="Homo sapiens[ORGN]", retmax=50, out_file="proteins.fasta"):
    handle = Entrez.esearch(db="protein", term=search_term, retmax=retmax)
    record = Entrez.read(handle)
    ids = record["IdList"]
    print(f"Found {len(ids)} protein IDs (retmax={retmax}).")

    fasta_seqs = []
    for seq_id in ids:
        fetch_handle = Entrez.efetch(db="protein", id=seq_id, rettype="fasta", retmode="text")
        fasta_record = fetch_handle.read()
        fasta_seqs.append(fasta_record)

    with open(out_file, "w") as f:
        f.writelines(fasta_seqs)

    print(f"Saved {len(fasta_seqs)} sequences to {out_file}")

if __name__ == "__main__":
    # Optional: pass search_term and retmax from CLI
    # e.g. python download_proteins.py "Homo sapiens[ORGN]" 100
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--term", default="Homo sapiens[ORGN]")
    p.add_argument("--retmax", type=int, default=50)
    p.add_argument("--out", default="proteins.fasta")
    args = p.parse_args()
    download_fasta(args.term, args.retmax, args.out)
