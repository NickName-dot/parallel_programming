import numpy as np
import sys
from pathlib import Path

def main():
    n = int(sys.argv[1])
    fileA = Path(sys.argv[2])
    fileB = Path(sys.argv[3])

    rng = np.random.default_rng(42)
    A = rng.integers(0, 10, size=(n, n), dtype=np.int32)
    B = rng.integers(0, 10, size=(n, n), dtype=np.int32)

    np.savetxt(fileA, A, fmt="%d", delimiter="\t")
    np.savetxt(fileB, B, fmt="%d", delimiter="\t")

if __name__ == "__main__":
    main()