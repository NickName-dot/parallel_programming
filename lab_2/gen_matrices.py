import numpy as np
import sys
from pathlib import Path

n = int(sys.argv[1])
fileA = Path(sys.argv[2])
fileB = Path(sys.argv[3])

rng = np.random.default_rng(42)
A = rng.integers(0, 256, size=(n, n), dtype=np.uint8)
B = rng.integers(0, 256, size=(n, n), dtype=np.uint8)

np.savetxt(fileA, A, fmt="%d", delimiter="\t")
np.savetxt(fileB, B, fmt="%d", delimiter="\t")