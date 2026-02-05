#include <iostream>
#include <vector>
#include <string>
#include <getopt.h>
#include <random>
#include <iomanip>
#include <chrono>
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
    TILED,
    TREE // CPU
} cudamode_t;

int main(int argc, char **argv)
{
    int n = 1024;
    size_t block_size = 256;
    int iterations = 10;
    int steps_per_frame = 20;
    cudamode_t mode = TILED;

    int opt;
    while ((opt = getopt(argc, argv, "n:s:i:m:b:")) != -1)
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
            else if (str_mode == "tree")
            {
                mode = TREE;
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
        case 'b':
            block_size = std::stoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -n <num_particles> -s <skip_nb_frame> -i <iterations> -m <naive|tiled|tree> -b <block_size>\n";
            return 1;
        }
    }

    std::vector<Particle> h_particles(n);
    std::mt19937 gen(42);

    // Orbital Velocity technique, making better dataset for better galaxy results
    constexpr FLOAT galaxy_radius = 100.0f;
    constexpr FLOAT core_radius = 5.0f;     // no particles spawned closer than this
    constexpr FLOAT disk_thickness = 2.0f;  // flatness of the galaxy
    constexpr FLOAT center_mass = 10000.0f; // mass of the black hole in the middle
    constexpr FLOAT G = 1.0f;               // gravity constant (simulation units)

    h_particles[0] = {0, 0, 0, 0, 0, 0, center_mass}; // black hole to center around

    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(core_radius, galaxy_radius);
    std::uniform_real_distribution<float> height_dist(-disk_thickness, disk_thickness);
    std::uniform_real_distribution<float> mass_dist(0.5f, 5.0f);

    for (int i = 1; i < n; ++i)
    {
        FLOAT angle = angle_dist(gen);
        FLOAT dist = radius_dist(gen);

        FLOAT px = dist * cos(angle);
        FLOAT py = dist * sin(angle);
        FLOAT pz = height_dist(gen);

        FLOAT velocity = sqrt(G * center_mass / dist);

        FLOAT vx = -velocity * sin(angle);
        FLOAT vy = velocity * cos(angle);
        FLOAT vz = 0.0f;

        FLOAT mass = mass_dist(gen);

        h_particles[i] = {px, py, pz, vx, vy, vz, mass};
    }

    // x, y, z, mass arrays, vx... aren't needed in calculations
    size_t shared_mem_size = 4 * block_size * sizeof(FLOAT);

    ParticleArrays d_particles;
    if (mode != TREE)
    {
        CUDA_CHECK(cudaMalloc(&d_particles.x, n * sizeof(FLOAT)));
        CUDA_CHECK(cudaMalloc(&d_particles.y, n * sizeof(FLOAT)));
        CUDA_CHECK(cudaMalloc(&d_particles.z, n * sizeof(FLOAT)));
        CUDA_CHECK(cudaMalloc(&d_particles.vx, n * sizeof(FLOAT)));
        CUDA_CHECK(cudaMalloc(&d_particles.vy, n * sizeof(FLOAT)));
        CUDA_CHECK(cudaMalloc(&d_particles.vz, n * sizeof(FLOAT)));
        CUDA_CHECK(cudaMalloc(&d_particles.mass, n * sizeof(FLOAT)));

        std::vector<FLOAT> temp(n);

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].x;
        CUDA_CHECK(cudaMemcpy(d_particles.x, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].y;
        CUDA_CHECK(cudaMemcpy(d_particles.y, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].z;
        CUDA_CHECK(cudaMemcpy(d_particles.z, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].vx;
        CUDA_CHECK(cudaMemcpy(d_particles.vx, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].vy;
        CUDA_CHECK(cudaMemcpy(d_particles.vy, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].vz;
        CUDA_CHECK(cudaMemcpy(d_particles.vz, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));

        for (int i = 0; i < n; ++i)
            temp[i] = h_particles[i].mass;
        CUDA_CHECK(cudaMemcpy(d_particles.mass, temp.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }

    FLOAT dt = 0.01f;
    FLOAT softening = 1e-9f;

    int num_blocks = (n + block_size - 1) / block_size;

    cudaEvent_t global_start, global_stop;
    std::chrono::steady_clock::time_point cpu_begin = std::chrono::steady_clock::now();
    if (mode != TREE)
    {
        CUDA_CHECK(cudaEventCreate(&global_start));
        CUDA_CHECK(cudaEventCreate(&global_stop));
        CUDA_CHECK(cudaEventRecord(global_start));
    }

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
            case TREE:
                nbody_barnes_hut_cpu(h_particles.data(), n, dt, softening);
                break;
            }
        }

        if (mode != TREE)
        {
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FLOAT> temp(n);

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.x, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].x = temp[i];

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.y, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].y = temp[i];

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.z, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].z = temp[i];

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.vx, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].vx = temp[i];

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.vy, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].vy = temp[i];

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.vz, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].vz = temp[i];

            CUDA_CHECK(cudaMemcpy(temp.data(), d_particles.mass, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_particles[i].mass = temp[i];
        }
        print_csv_row(h_particles);
    }

    float elapsed_ms = 0.0f;
    if (mode != TREE)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(global_stop));
        CUDA_CHECK(cudaEventSynchronize(global_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, global_start, global_stop));
    }
    else
    {
        std::chrono::steady_clock::time_point cpu_end = std::chrono::steady_clock::now();
        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_begin).count();
    }
    std::cerr << "Total elapsed (s): " << (elapsed_ms / 1000.0f) << "\n";

    if (mode != TREE)
    {
        CUDA_CHECK(cudaEventDestroy(global_start));
        CUDA_CHECK(cudaEventDestroy(global_stop));

        CUDA_CHECK(cudaFree(d_particles.x));
        CUDA_CHECK(cudaFree(d_particles.y));
        CUDA_CHECK(cudaFree(d_particles.z));
        CUDA_CHECK(cudaFree(d_particles.vx));
        CUDA_CHECK(cudaFree(d_particles.vy));
        CUDA_CHECK(cudaFree(d_particles.vz));
        CUDA_CHECK(cudaFree(d_particles.mass));
    }
    return 0;
}