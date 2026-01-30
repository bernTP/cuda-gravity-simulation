#include "nbody.cuh"
#include <cmath>

/* Using shared memory, maybe there are better ways */

__global__ void nbody_tiled_kernel(Particle *particles, int n, float dt, float softening)
{
    extern __shared__ Particle shared_particles[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

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
                float dx = shared_particles[j].x - pi.x;
                float dy = shared_particles[j].y - pi.y;
                float dz = shared_particles[j].z - pi.z;

                float distSqr = dx * dx + dy * dy + dz * dz + softening;
                float invDist = 1.0f / sqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                float s = shared_particles[j].mass * invDist3;

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
