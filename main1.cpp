#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    for (int N : sizes)
    {
        std::vector<long long> A, B, C;
        if (rank == 0)
        {
            A.resize(N * N);
            B.resize(N * N);
            C.resize(N * N, 0);

            std::srand(static_cast<unsigned>(std::time(nullptr)) + N);
            for (int i = 0; i < N * N; ++i)
            {
                A[i] = std::rand() % 100;
                B[i] = std::rand() % 100;
            }
        }

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0)
            B.resize(N * N);

        MPI_Bcast(B.data(), N * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        int rows_per_proc = N / size;
        int remainder = N % size;
        int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

        std::vector<int> sendcounts(size), displs(size);
        int offset = 0;
        for (int i = 0; i < size; ++i)
        {
            int r = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = r * N;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        std::vector<long long> local_A(local_rows * N);
        std::vector<long long> local_C(local_rows * N, 0);

        MPI_Scatterv(
            rank == 0 ? A.data() : nullptr,
            sendcounts.data(),
            displs.data(),
            MPI_LONG_LONG,
            local_A.data(),
            local_rows * N,
            MPI_LONG_LONG,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        for (int i = 0; i < local_rows; ++i)
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j)
                    local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];

        double local_time = MPI_Wtime() - start_time;
        double max_time = 0.0;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Gatherv(
            local_C.data(),
            local_rows * N,
            MPI_LONG_LONG,
            rank == 0 ? C.data() : nullptr,
            sendcounts.data(),
            displs.data(),
            MPI_LONG_LONG,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            std::cout << "N=" << N
                      << " processes=" << size
                      << " time=" << std::fixed << std::setprecision(7)
                      << max_time
                      << " seconds" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}