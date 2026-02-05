#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

#define FLOAT float

struct ParticleArrays {
    FLOAT *x, *y, *z;
    FLOAT *vx, *vy, *vz;
    FLOAT *mass;
};

// struct for CPU-side initialization
// due to bad coalescing, nontheless the performance gain from avoiding this struct was very minimal
// principal bottleneck here is most likely calculation
struct Particle {
    FLOAT x, y, z;
    FLOAT vx, vy, vz;
    FLOAT mass;
};

__global__ void nbody_naive_kernel(ParticleArrays particles, int n, FLOAT dt, FLOAT softening);
__global__ void nbody_tiled_kernel(ParticleArrays particles, int n, FLOAT dt, FLOAT softening);
void nbody_barnes_hut_cpu(Particle *particles, int n, FLOAT dt, FLOAT softening);

#endif // NBODY_CUH
