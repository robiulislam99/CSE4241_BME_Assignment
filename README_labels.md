README_labels.md

IMPORTANT:
NCBI FASTA sequences do NOT include secondary structure (2D) annotations.
You need per-residue labels (one label per amino acid) in the same order as sequence.

Options to obtain labels:
1) Download Dataset:
   - CB513, CullPDB, or ProteinNet contain sequences + secondary structure labels.
   - Example sources (search web): "CB513 secondary structure dataset", "ProteinNet secondary structure".

2) Derive labels from PDB:
   - Download PDB structure files and run DSSP to extract secondary structure per residue.
   - Tools: DSSP (mkdssp), BioPython's DSSP module. This requires PDB <-> sequence mapping.

Format expected by this project:
- Two files (aligned by index):
  - sequences.fasta  (each record is a sequence in FASTA)
  - labels.txt       (each line corresponds to one sequence; characters 'H', 'E', 'C' per residue)
    e.g. for a sequence of length 10: "CCCCCHHHHE"
