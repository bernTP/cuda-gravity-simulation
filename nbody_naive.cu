#include "nbody.cuh"

__global__ void nbody_naive_kernel(ParticleArrays p, int n, FLOAT dt, FLOAT softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    FLOAT fx = 0.0f;
    FLOAT fy = 0.0f;
    FLOAT fz = 0.0f;

    FLOAT px = p.x[i];
    FLOAT py = p.y[i];
    FLOAT pz = p.z[i];

    for (int j = 0; j < n; j++)
    {
        FLOAT dx = p.x[j] - px;
        FLOAT dy = p.y[j] - py;
        FLOAT dz = p.z[j] - pz;

        FLOAT distSqr = dx * dx + dy * dy + dz * dz + softening;
        FLOAT invDist = 1.0f / rsqrtf(distSqr);
        FLOAT invDist3 = invDist * invDist * invDist;

        FLOAT s = p.mass[j] * invDist3;

        fx += dx * s;
        fy += dy * s;
        fz += dz * s;
    }

    p.vx[i] += dt * fx;
    p.vy[i] += dt * fy;
    p.vz[i] += dt * fz;

    p.x[i] += dt * p.vx[i];
    p.y[i] += dt * p.vy[i];
    p.z[i] += dt * p.vz[i];
}
