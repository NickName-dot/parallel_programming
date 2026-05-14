import csv
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"C:\Users\Nikita\Desktop\parallel_programming\lab_2")
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8]

CPP_EXE = BASE / "lab2.exe"
GEN_SCRIPT = BASE / "gen_matrices.py"
VERIFY_SCRIPT = BASE / "verify.py"

def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, check=True, **kwargs)

def main():
    rows = []

    for n in SIZES:
        a_file = OUT_DIR / f"A_{n}.txt"
        b_file = OUT_DIR / f"B_{n}.txt"

        run([sys.executable, str(GEN_SCRIPT), str(n), str(a_file), str(b_file)], cwd=BASE)

        for t in THREADS:
            start = time.perf_counter()

            pgen = subprocess.run(
                [sys.executable, str(GEN_SCRIPT), str(n), str(a_file), str(b_file)],
                cwd=BASE,
                text=True,
                capture_output=True
            )
            if pgen.returncode != 0:
                print("GEN ERROR:")
                print(pgen.stdout)
                print(pgen.stderr)
                raise SystemExit(1)
            [str(CPP_EXE)],

            elapsed = time.perf_counter() - start
            cpp_output = BASE / f"result_{n}x{n}_t{t}.txt"

            verify = subprocess.run(
                [sys.executable, str(VERIFY_SCRIPT), str(a_file), str(b_file), str(cpp_output)],
                text=True,
                capture_output=True,
                cwd=BASE
            )

            rows.append({
                "n": n,
                "threads": t,
                "time_sec": f"{elapsed:.6f}",
                "cpp_returncode": pgen.returncode,
                "verify_stdout": verify.stdout.strip(),
                "verify_returncode": verify.returncode,
                "cpp_stdout": pgen.stdout.strip(),
                "cpp_stderr": pgen.stderr.strip(),
            })

            print(f"Перемножение матриц {n}x{n} с кол-вом ядер {t} завершено. Время: {elapsed:.6f} сек. Проверка: {verify.stdout.strip()}", flush=True)

    csv_file = OUT_DIR / "benchmark_results.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Сохранён CSV: {csv_file}", flush=True)

if __name__ == "__main__":
    main()