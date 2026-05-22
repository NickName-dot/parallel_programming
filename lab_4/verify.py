import sys
from pathlib import Path
import numpy as np

def main():
    fileA = Path(sys.argv[1])
    fileB = Path(sys.argv[2])
    fileC = Path(sys.argv[3])

    A = np.loadtxt(fileA, dtype=np.float32, delimiter="\t")
    B = np.loadtxt(fileB, dtype=np.float32, delimiter="\t")
    C = np.loadtxt(fileC, dtype=np.float32, delimiter="\t")

    expected = A @ B

    if np.allclose(expected, C):
        print("SUCCESS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()