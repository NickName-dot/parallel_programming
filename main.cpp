#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstdint>
#include <chrono>
 
using Matrix = std::vector<std::vector<uint8_t>>;
using Clock = std::chrono::high_resolution_clock;

Matrix read_square_matrix(const std::string& filename, size_t& size) {
    Matrix matrix;
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Error: cannot open " << filename << std::endl; return {}; }
    
    std::string line; size = 0;
    while (std::getline(file, line)) {
        std::vector<uint8_t> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) {
            row.push_back(static_cast<uint8_t>(std::stoul(cell)));
        }
        if (!row.empty()) {
            if (!matrix.empty() && row.size() != matrix[0].size()) {
                std::cerr << "Error: matrix is not square!" << std::endl; return {};
            }
            matrix.push_back(row); size++;
        }
    }
    file.close();
    
    if (matrix.empty() || matrix.size() != matrix[0].size()) {
        std::cerr << "Error: " << filename << " is not square!" << std::endl; return {};
    }
    return matrix;
}

Matrix multiply_square_matrices(const Matrix& A, const Matrix& B, size_t size) {
    Matrix C(size, std::vector<uint8_t>(size, 0));
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            uint32_t sum = 0;
            for (size_t k = 0; k < size; ++k) {
                sum += static_cast<uint32_t>(A[i][k]) * B[k][j];
            }
            C[i][j] = static_cast<uint8_t>(sum % 256);
        }
    }
    return C;
}

void save_result(const Matrix& matrix, size_t size, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) { std::cerr << "Error saving " << filename << std::endl; return; }
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << static_cast<int>(row[i]);
            if (i < row.size() - 1) file << "\t";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Result saved: " << filename << std::endl;
}

int main() {
    std::string fileA, fileB;
    std::cout << "File for first matrix: "; std::cin >> fileA;
    std::cout << "File for second matrix: "; std::cin >> fileB;
    
    size_t sizeA, sizeB;
    Matrix A = read_square_matrix(fileA, sizeA);
    Matrix B = read_square_matrix(fileB, sizeB);
    
    if (A.empty() || B.empty() || sizeA != sizeB) {
        std::cerr << "Error: invalid matrices!" << std::endl; return 1;
    }
    
    size_t n = sizeA;
    double flops = 2.0 * n * n * n / 1e9;
    std::cout << "\nTask size: n=" << n << ", operations: " 
              << std::fixed << std::setprecision(2) << 2.0 * n * n * n << " (" << flops << " GFLOPS)" << std::endl;
    
    auto start = Clock::now();
    Matrix C = multiply_square_matrices(A, B, n);
    auto end = Clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    
    std::cout << "\nExecution time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    
    std::string output_file = "result_" + std::to_string(n) + "x" + std::to_string(n) + ".txt";
    save_result(C, n, output_file);
    std::cout << "\nDone!" << std::endl;
    return 0;
}
