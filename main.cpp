#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdint>
#include <chrono>

using Matrix = std::vector<std::vector<uint8_t>>;

Matrix read_square_matrix(const std::string& filename, size_t& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return {};
    }

    Matrix matrix;
    std::string line;
    n = 0;

    while (std::getline(file, line)) {
        std::vector<uint8_t> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) {
            if (!cell.empty()) row.push_back(static_cast<uint8_t>(std::stoul(cell)));
        }
        if (!row.empty()) {
            if (!matrix.empty() && row.size() != matrix[0].size()) {
                std::cerr << "Error: matrix is not rectangular!" << std::endl;
                return {};
            }
            matrix.push_back(row);
            ++n;
        }
    }

    if (matrix.empty() || matrix.size() != matrix[0].size()) {
        std::cerr << "Error: " << filename << " is not square!" << std::endl;
        return {};
    }

    return matrix;
}

void save_result(const Matrix& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error saving " << filename << std::endl;
        return;
    }

    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << static_cast<unsigned>(row[i]);
            if (i + 1 < row.size()) file << '\t';
        }
        file << '\n';
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nA = 0, nB = 0, n = 0;
    Matrix A, B;

    if (rank == 0) {
        std::string fileA, fileB;
        std::cout << "File for first matrix: ";
        std::cin >> fileA;
        std::cout << "File for second matrix: ";
        std::cin >> fileB;

        A = read_square_matrix(fileA, nA);
        B = read_square_matrix(fileB, nB);

        if (A.empty() || B.empty() || nA != nB || A.size() != B.size()) {
            std::cerr << "Error: invalid matrices!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = nA;
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.assign(n, std::vector<uint8_t>(n));
    }

    std::vector<uint8_t> flatB(n * n);
    if (rank == 0) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                flatB[i * n + j] = B[i][j];
    }

    MPI_Bcast(flatB.data(), static_cast<int>(n * n), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                B[i][j] = flatB[i * n + j];
    }

    int rows_per_proc = static_cast<int>(n) / size;
    int remainder = static_cast<int>(n) % size;

    std::vector<int> sendcounts(size), displs(size);
    for (int p = 0; p < size; ++p) {
        int rows = rows_per_proc + (p < remainder ? 1 : 0);
        sendcounts[p] = rows * static_cast<int>(n);
        displs[p] = (p == 0 ? 0 : displs[p - 1] + sendcounts[p - 1]);
    }

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    std::vector<uint8_t> Aflat, A_local(local_rows * n), C_local(local_rows * n), Cflat;

    if (rank == 0) {
        Aflat.resize(n * n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                Aflat[i * n + j] = A[i][j];
    }

    MPI_Scatterv(
        rank == 0 ? Aflat.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_UNSIGNED_CHAR,
        A_local.data(),
        static_cast<int>(A_local.size()),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uint32_t sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += static_cast<uint32_t>(A_local[i * n + k]) * static_cast<uint32_t>(flatB[k * n + j]);
            }
            C_local[i * n + j] = static_cast<uint8_t>(sum % 256);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double local_time = std::chrono::duration<double, std::milli>(end - start).count();

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Cflat.resize(n * n);
    }

    MPI_Gatherv(
        C_local.data(),
        static_cast<int>(C_local.size()),
        MPI_UNSIGNED_CHAR,
        rank == 0 ? Cflat.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        Matrix C(n, std::vector<uint8_t>(n));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                C[i][j] = Cflat[i * n + j];

        std::cout << "\nTask size: n=" << n
                  << ", processes=" << size
                  << ", execution time: " << std::fixed << std::setprecision(3)
                  << max_time << " ms" << std::endl;

        std::string output_file = "result_" + std::to_string(n) + "x" + std::to_string(n) + "_p" + std::to_string(size) + ".txt";
        save_result(C, output_file);
        std::cout << "Result saved: " << output_file << std::endl;
    }

    MPI_Finalize();
    return 0;
}