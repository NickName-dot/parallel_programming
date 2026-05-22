#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

using Matrix = std::vector<float>;

Matrix read_matrix(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<float>> rows;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) {
            if (!cell.empty()) row.push_back(std::stof(cell));
        }
        if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty() || rows.size() != rows[0].size()) {
        std::cerr << "Error: " << filename << " is not square!" << std::endl;
        return {};
    }

    n = static_cast<int>(rows.size());
    Matrix m(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m[i * n + j] = rows[i][j];

    return m;
}

void save_matrix(const Matrix& m, int n, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error saving " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << std::fixed << std::setprecision(0) << m[i * n + j];
            if (j + 1 < n) file << '\t';
        }
        file << '\n';
    }
}

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: cuda_mmul.exe <N> <block_x> <block_y>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    int block_x = std::stoi(argv[2]);
    int block_y = std::stoi(argv[3]);

    std::string fileA, fileB;
    std::cout << "File for first matrix: ";
    std::cin >> fileA;
    std::cout << "File for second matrix: ";
    std::cin >> fileB;

    int nA = 0, nB = 0;
    Matrix A = read_matrix(fileA, nA);
    Matrix B = read_matrix(fileB, nB);

    if (A.empty() || B.empty() || nA != nB || nA != n) {
        std::cerr << "Error: invalid matrices or size mismatch!" << std::endl;
        return 1;
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    Matrix C(n * n, 0.0f);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    CHECK_CUDA(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(block_x, block_y);
    dim3 grid((n + block_x - 1) / block_x, (n + block_y - 1) / block_y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    std::string output_file = "result_" + std::to_string(n) + "_b" +
                              std::to_string(block_x) + "x" + std::to_string(block_y) + ".txt";
    save_matrix(C, n, output_file);

    std::cout << "\nTask size: n=" << n
              << ", block=" << block_x << "x" << block_y
              << ", execution time: " << std::fixed << std::setprecision(3)
              << elapsed_ms << " ms" << std::endl;

    std::cout << "Result saved: " << output_file << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}