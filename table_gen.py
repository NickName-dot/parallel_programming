import numpy as np

def generate_square_matrix(size: int, dtype=np.uint8) -> np.ndarray:
    return np.random.randint(0, 256, size=(size, size), dtype=dtype)

def save_matrix_to_txt(matrix: np.ndarray, filename: str):
    np.savetxt(filename, matrix, fmt='%d', delimiter='\t')

n = int(input("Размер матрицы (n x n): "))
matrix = generate_square_matrix(n)

print(f"\nMatrix {n}x{n} ({matrix.nbytes} bytes):")
print(matrix)

filename = input("Имя файла (Enter для matrix1.txt): ").strip() or "matrix1.txt"
save_matrix_to_txt(matrix, filename)
print(f"Saved to {filename}")
