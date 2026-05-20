import sys
from pathlib import Path
import numpy as np

def load_matrix(path):
    return np.loadtxt(path, dtype=np.uint8, delimiter="\t")

def main():
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    C = load_matrix(input_file)

    total = int(C.sum())
    mn = int(C.min())
    mx = int(C.max())
    avg = float(C.mean())

    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"File: {input_file.name}\n")
        f.write(f"Size: {C.shape[0]}x{C.shape[1]}\n")
        f.write(f"Sum: {total}\n")
        f.write(f"Min: {mn}\n")
        f.write(f"Max: {mx}\n")
        f.write(f"Mean: {avg:.6f}\n")

if __name__ == "__main__":
    main()