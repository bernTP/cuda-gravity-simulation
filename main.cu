#include <iostream>
#include <vector>
#include <string>
#include <getopt.h>
#include <random>
#include <iomanip>
#include <cmath> // Required for sin, cos, sqrt
#include "nbody.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

typedef enum CudaMode
{
    NAIVE,
    TILED
} cudamode_t;

int main(int argc, char **argv)
{
    int n = 1024;
    constexpr size_t block_size = 256;
    int iterations = 10;
    int steps_per_frame = 20;
    cudamode_t mode = TILED;
    constexpr size_t shared_mem_size = block_size * sizeof(Particle);

    int opt;
    while ((opt = getopt(argc, argv, "n:s:i:m:")) != -1)
    {
        switch (opt)
        {
        case 'n':
            n = std::stoi(optarg);
            break;
        case 'i':
            iterations = std::stoi(optarg);
            break;
        case 'm':
        {
            std::string str_mode = optarg;
            if (str_mode == "naive")
            {
                mode = NAIVE;
            }
            else if (str_mode == "tiled")
            {
                mode = TILED;
            }
            else
            {
                std::cerr << "Unknown mode: " << str_mode << "\n";
                return 1;
            }
            break;
        }
        case 's':
            steps_per_frame = std::stoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -n <num_particles> -s <skip_nb_frame> -i <iterations> -m <naive|tiled>\n";
            return 1;
        }
    }

    std::vector<Particle> h_particles(n);
    std::mt19937 gen(42);

    // Orbital Velocity technique, making better dataset for better galaxy results
    constexpr float galaxy_radius = 100.0f;
    constexpr float core_radius = 5.0f;     // no particles spawned closer than this
    constexpr float disk_thickness = 2.0f;  // flatness of the galaxy
    constexpr float center_mass = 10000.0f; // mass of the black hole in the middle
    constexpr float G = 1.0f;               // gravity constant (simulation units)

    h_particles[0] = {0, 0, 0, 0, 0, 0, center_mass}; // black hole to center around

    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(core_radius, galaxy_radius);
    std::uniform_real_distribution<float> height_dist(-disk_thickness, disk_thickness);
    std::uniform_real_distribution<float> mass_dist(0.5f, 5.0f);

    for (int i = 1; i < n; ++i)
    {
        float angle = angle_dist(gen);
        float dist = radius_dist(gen);

        float px = dist * cos(angle);
        float py = dist * sin(angle);
        float pz = height_dist(gen);

        float velocity = sqrt(G * center_mass / dist);

        float vx = -velocity * sin(angle);
        float vy = velocity * cos(angle);
        float vz = 0.0f;

        float mass = mass_dist(gen);

        h_particles[i] = {px, py, pz, vx, vy, vz, mass};
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
            switch (mode)
            {
            case NAIVE:
                nbody_naive_kernel<<<num_blocks, block_size>>>(d_particles, n, dt, softening);
                break;
            case TILED:
                nbody_tiled_kernel<<<num_blocks, block_size, shared_mem_size>>>(d_particles, n, dt, softening);
                break;
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