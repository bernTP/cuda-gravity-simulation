#include <iostream>
#include <vector>
#include <string>
#include <getopt.h>
#include <random>
#include <iomanip>
#include "nbody.cuh"

#define CUDA_CHECK(call)                                                                                                  \
    do                                                                                                                    \
    {                                                                                                                     \
        cudaError_t err = call;                                                                                           \
        if (err != cudaSuccess)                                                                                           \
        {                                                                                                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                                                                      \
        }                                                                                                                 \
    } while (0)

void print_csv_row(const std::vector<Particle> &particles)
{
    for (size_t i = 0; i < particles.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(8) << particles[i].x << "," << particles[i].y << "," << particles[i].z;
        if (i < particles.size() - 1)
            std::cout << ",";
    }
    std::cout << "\n";
}

int main(int argc, char **argv)
{
    int n = 1024;
    int block_size = 256;
    int iterations = 10;
    int steps_per_frame = 20;
    std::string mode = "naive";

    int opt;
    while ((opt = getopt(argc, argv, "n:s:b:i:m:")) != -1)
    {
        switch (opt)
        {
        case 'n':
            n = std::stoi(optarg);
            break;
        case 'b':
            block_size = std::stoi(optarg);
            break;
        case 'i':
            iterations = std::stoi(optarg);
            break;
        case 'm':
            mode = optarg;
            break;
        case 's':
            steps_per_frame = std::stoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -n <num_particles> -b <block_size> -i <iterations> -m <naive|tiled>\n";
            return 1;
        }
    }

    std::vector<Particle> h_particles(n);
    std::mt19937 gen(42); // random
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f); // random values between these values (hard coded)
    std::uniform_real_distribution<float> mass_dist(1.0f, 10.0f);

    for (int i = 0; i < n; ++i)
    {
        h_particles[i] = {dist(gen), dist(gen), dist(gen), 0, 0, 0, mass_dist(gen)};
    }

    Particle *d_particles;
    CUDA_CHECK(cudaMalloc(&d_particles, n * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice));

    float dt = 0.01f;
    float softening = 1e-9f;

    int num_blocks = (n + block_size - 1) / block_size;

    cudaEvent_t global_start, global_stop;
    CUDA_CHECK(cudaEventCreate(&global_start));
    CUDA_CHECK(cudaEventCreate(&global_stop));
    CUDA_CHECK(cudaEventRecord(global_start));

    for (int iter = 0; iter < iterations; iter += steps_per_frame)
    {
        for (int s = 0; s < steps_per_frame; ++s)
        {
            if (mode == "naive")
            {
                nbody_naive_kernel<<<num_blocks, block_size>>>(d_particles, n, dt, softening);
            }
            else
            {
                size_t shared_mem_size = block_size * sizeof(Particle);
                nbody_tiled_kernel<<<num_blocks, block_size, shared_mem_size>>>(d_particles, n, dt, softening);
            }
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost));
        print_csv_row(h_particles);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(global_stop));
    CUDA_CHECK(cudaEventSynchronize(global_stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, global_start, global_stop));
    std::cerr << "Total elapsed (s): " << (elapsed_ms / 1000.0f) << "\n";

    CUDA_CHECK(cudaEventDestroy(global_start));
    CUDA_CHECK(cudaEventDestroy(global_stop));

    CUDA_CHECK(cudaFree(d_particles));
    return 0;
}