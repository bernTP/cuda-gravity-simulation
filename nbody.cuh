#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

__global__ void nbody_naive_kernel(Particle* particles, int n, float dt, float softening);
__global__ void nbody_tiled_kernel(Particle* particles, int n, float dt, float softening);

#endif // NBODY_CUH
