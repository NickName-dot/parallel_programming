import csv
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/Nikita/Desktop/parallel_programming/lab_3")
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS = [1, 2, 4, 8]

MPI_EXE = BASE / "lab_mpi.exe"
GEN_SCRIPT = BASE / "gen_matrices.py"
VERIFY_SCRIPT = BASE / "verify.py"
PROCESS_SCRIPT = BASE / "process_result.py"

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

        for p in PROCS:
            start = time.perf_counter()

            mpiexec_cmd = [
                "mpiexec",
                "-n",
                str(p),
                str(MPI_EXE)
            ]

            proc = subprocess.run(
                mpiexec_cmd,
                input=f"{a_file}\n{b_file}\n",
                text=True,
                capture_output=True,
                cwd=BASE
            )

            elapsed = time.perf_counter() - start

            cpp_output = BASE / f"result_{n}x{n}_p{p}.txt"
            analysis_output = BASE / f"analysis_{n}x{n}_p{p}.txt"

            verify = run([sys.executable, str(VERIFY_SCRIPT), str(a_file), str(b_file), str(cpp_output)], cwd=BASE)

            if cpp_output.exists():
                process = run([sys.executable, str(PROCESS_SCRIPT), str(cpp_output), str(analysis_output)], cwd=BASE)
                process_rc = process.returncode
            else:
                process_rc = 1

            rows.append({
                "n": n,
                "processes": p,
                "time_sec": f"{elapsed:.6f}",
                "mpi_returncode": proc.returncode,
                "verify_stdout": verify.stdout.strip(),
                "verify_returncode": verify.returncode,
                "process_returncode": process_rc,
                "mpi_stdout": proc.stdout.strip(),
                "mpi_stderr": proc.stderr.strip(),
            })

            print(
                f"Перемножение матриц {n}x{n} с кол-вом процессов {p} завершено. "
                f"Время: {elapsed:.6f} сек. Проверка: {verify.stdout.strip()}",
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