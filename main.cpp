#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <omp.h>

using Matrix = std::vector<std::vector<uint8_t>>;
using Clock = std::chrono::high_resolution_clock;

Matrix read_square_matrix(const std::string& filename, size_t& size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return {};
    }

    Matrix matrix;
    std::string line;
    size = 0;

    while (std::getline(file, line)) {
        std::vector<uint8_t> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, '\t')) {
            if (!cell.empty()) {
                row.push_back(static_cast<uint8_t>(std::stoul(cell)));
            }
        }

        if (!row.empty()) {
            if (!matrix.empty() && row.size() != matrix[0].size()) {
                std::cerr << "Error: matrix is not rectangular!" << std::endl;
                return {};
            }
            matrix.push_back(row);
            ++size;
        }
    }

    if (matrix.empty() || matrix.size() != matrix[0].size()) {
        std::cerr << "Error: " << filename << " is not square!" << std::endl;
        return {};
    }

    return matrix;
}

Matrix multiply_square_matrices_omp(const Matrix& A, const Matrix& B, size_t n, int threads) {
    Matrix C(n, std::vector<uint8_t>(n, 0));

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        for (int j = 0; j < static_cast<int>(n); ++j) {
            uint32_t sum = 0;
            for (int k = 0; k < static_cast<int>(n); ++k) {
                sum += static_cast<uint32_t>(A[i][k]) * static_cast<uint32_t>(B[k][j]);
            }
            C[i][j] = static_cast<uint8_t>(sum % 256);
        }
    }

    return C;
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

int main() {
    std::string fileA, fileB;
    int threads = 1;

    std::cout << "File for first matrix: ";
    std::cin >> fileA;

    std::cout << "File for second matrix: ";
    std::cin >> fileB;

    std::cout << "Threads: ";
    std::cin >> threads;

    if (threads < 1) threads = 1;

    size_t sizeA = 0, sizeB = 0;
    Matrix A = read_square_matrix(fileA, sizeA);
    Matrix B = read_square_matrix(fileB, sizeB);

    if (A.empty() || B.empty() || sizeA != sizeB) {
        std::cerr << "Error: invalid matrices!" << std::endl;
        return 1;
    }

    size_t n = sizeA;
    std::cout << "\nTask size: n=" << n
              << ", threads=" << threads
              << ", operations=" << std::fixed << std::setprecision(2)
              << 2.0 * n * n * n << std::endl;

    auto start = Clock::now();
    Matrix C = multiply_square_matrices_omp(A, B, n, threads);
    auto end = Clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Execution time: " << std::fixed << std::setprecision(3)
              << time_ms << " ms" << std::endl;

    std::string output_file = "result_" + std::to_string(n) + "x" + std::to_string(n)
                            + "_t" + std::to_string(threads) + ".txt";
    save_result(C, output_file);

    std::cout << "Result saved: " << output_file << std::endl;
    return 0;
}