import sys
from pathlib import Path
import numpy as np

def main():
    fileA = Path(sys.argv[1])
    fileB = Path(sys.argv[2])
    fileC = Path(sys.argv[3])

    A = np.loadtxt(fileA, dtype=np.uint8, delimiter="\t")
    B = np.loadtxt(fileB, dtype=np.uint8, delimiter="\t")
    C = np.loadtxt(fileC, dtype=np.uint8, delimiter="\t")

    expected = (A.astype(np.uint32) @ B.astype(np.uint32)) % 256

    if np.array_equal(expected.astype(np.uint8), C.astype(np.uint8)):
        print("SUCCESS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()