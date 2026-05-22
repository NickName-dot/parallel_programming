import csv
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/Nikita/Desktop/parallel_programming/lab_4")
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

SIZES = [200, 400, 800, 1200, 1600, 2000]
BLOCKS = [(8, 8), (16, 16), (32, 32)]

CUDA_EXE = BASE / "cuda_mmul.exe"
GEN_SCRIPT = BASE / "gen_matrices.py"
VERIFY_SCRIPT = BASE / "verify.py"

def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)

def main():
    rows = []

    for n in SIZES:
        a_file = OUT_DIR / f"A_{n}.txt"
        b_file = OUT_DIR / f"B_{n}.txt"

        gen = run([sys.executable, str(GEN_SCRIPT), str(n), str(a_file), str(b_file)], cwd=BASE)
        if gen.returncode != 0:
            print("Generator error:")
            print(gen.stdout)
            print(gen.stderr)
            raise SystemExit(1)

        for bx, by in BLOCKS:
            start = time.perf_counter()

            proc = subprocess.run(
                [str(CUDA_EXE), str(n), str(bx), str(by)],
                input=f"{a_file}\n{b_file}\n",
                text=True,
                capture_output=True,
                cwd=BASE
            )

            elapsed = time.perf_counter() - start

            cpp_output = BASE / f"result_{n}_b{bx}x{by}.txt"
            verify = run([sys.executable, str(VERIFY_SCRIPT), str(a_file), str(b_file), str(cpp_output)], cwd=BASE)

            rows.append({
                "n": n,
                "block_x": bx,
                "block_y": by,
                "time_sec": f"{elapsed:.6f}",
                "returncode": proc.returncode,
                "verify_stdout": verify.stdout.strip(),
                "verify_returncode": verify.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            })

            print(
                f"Матрица {n}x{n}, блок {bx}x{by}: "
                f"{elapsed:.6f} сек, проверка: {verify.stdout.strip()}",
                flush=True
            )

    csv_file = OUT_DIR / "benchmark_results.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Сохранён CSV: {csv_file}", flush=True)

if __name__ == "__main__":
    main()