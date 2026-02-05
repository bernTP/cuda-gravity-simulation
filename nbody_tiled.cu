#include "nbody.cuh"
#include <cmath>

/* Using shared memory with SoA layout for better coalescing */

__global__ void nbody_tiled_kernel(ParticleArrays p, int n, FLOAT dt, FLOAT softening)
{
    extern __shared__ FLOAT shared_mem[];

    FLOAT *s_x = &shared_mem[0 * blockDim.x];
    FLOAT *s_y = &shared_mem[1 * blockDim.x];
    FLOAT *s_z = &shared_mem[2 * blockDim.x];
    FLOAT *s_mass = &shared_mem[3 * blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT fx = 0.0f;
    FLOAT fy = 0.0f;
    FLOAT fz = 0.0f;

    FLOAT px, py, pz;
    if (i < n)
    {
        px = p.x[i];
        py = p.y[i];
        pz = p.z[i];
    }

    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; tile++)
    {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            s_x[threadIdx.x] = p.x[idx];
            s_y[threadIdx.x] = p.y[idx];
            s_z[threadIdx.x] = p.z[idx];
            s_mass[threadIdx.x] = p.mass[idx];
        }
        else
        {
            s_x[threadIdx.x] = 0.0f;
            s_y[threadIdx.x] = 0.0f;
            s_z[threadIdx.x] = 0.0f;
            s_mass[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (i < n)
        {
            for (int j = 0; j < blockDim.x; j++)
            {
                FLOAT dx = s_x[j] - px;
                FLOAT dy = s_y[j] - py;
                FLOAT dz = s_z[j] - pz;

                FLOAT distSqr = dx * dx + dy * dy + dz * dz + softening;
                FLOAT invDist = 1.0f / sqrtf(distSqr);
                FLOAT invDist3 = invDist * invDist * invDist;

                FLOAT s = s_mass[j] * invDist3;

                fx += dx * s;
                fy += dy * s;
                fz += dz * s;
            }
        }
        __syncthreads();
    }

    if (i < n)
    {
        p.vx[i] += dt * fx;
        p.vy[i] += dt * fy;
        p.vz[i] += dt * fz;

        p.x[i] += dt * p.vx[i];
        p.y[i] += dt * p.vy[i];
        p.z[i] += dt * p.vz[i];
    }
}
