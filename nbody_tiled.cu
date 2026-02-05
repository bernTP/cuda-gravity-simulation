#include "nbody.cuh"
#include <cmath>

/* Using shared memory, maybe there are better ways */

__global__ void nbody_tiled_kernel(Particle *particles, int n, FLOAT dt, FLOAT softening)
{
    extern __shared__ Particle shared_particles[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT fx = 0.0f;
    FLOAT fy = 0.0f;
    FLOAT fz = 0.0f;

    Particle pi;
    if (i < n)
    {
        pi = particles[i];
    }

    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; tile++)
    {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            shared_particles[threadIdx.x] = particles[idx];
        }
        else
        {
            shared_particles[threadIdx.x] = {0, 0, 0, 0, 0, 0, 0};
        }
        __syncthreads();

        if (i < n)
        {
            for (int j = 0; j < blockDim.x; j++)
            {
                FLOAT dx = shared_particles[j].x - pi.x;
                FLOAT dy = shared_particles[j].y - pi.y;
                FLOAT dz = shared_particles[j].z - pi.z;

                FLOAT distSqr = dx * dx + dy * dy + dz * dz + softening;
                FLOAT invDist = 1.0f / sqrtf(distSqr);
                FLOAT invDist3 = invDist * invDist * invDist;

                FLOAT s = shared_particles[j].mass * invDist3;

                fx += dx * s;
                fy += dy * s;
                fz += dz * s;
            }
        }
        __syncthreads();
    }

    if (i < n)
    {
        particles[i].vx += dt * fx;
        particles[i].vy += dt * fy;
        particles[i].vz += dt * fz;

        particles[i].x += dt * particles[i].vx;
        particles[i].y += dt * particles[i].vy;
        particles[i].z += dt * particles[i].vz;
    }
}
