import numpy as np
import sys

A = np.loadtxt(sys.argv[1], dtype=np.uint8, delimiter='\t')
B = np.loadtxt(sys.argv[2], dtype=np.uint8, delimiter='\t')
C_cpp = np.loadtxt(sys.argv[3], dtype=np.uint8, delimiter='\t')

C_ref = np.dot(A.astype(np.int32), B.astype(np.int32)) % 256

print("OK" if np.array_equal(C_cpp, C_ref) else "ERROR")
