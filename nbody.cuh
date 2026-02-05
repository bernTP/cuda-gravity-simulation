#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

#define FLOAT float

struct Particle {
    FLOAT x, y, z;
    FLOAT vx, vy, vz;
    FLOAT mass;
};


__global__ void nbody_naive_kernel(Particle* particles, int n, FLOAT dt, FLOAT softening);
__global__ void nbody_tiled_kernel(Particle* particles, int n, FLOAT dt, FLOAT softening);
void nbody_barnes_hut_cpu(Particle *particles, int n, FLOAT dt, FLOAT softening);

#endif // NBODY_CUH
