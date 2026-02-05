#include "nbody.cuh"

__global__ void nbody_naive_kernel(Particle *particles, int n, FLOAT dt, FLOAT softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    FLOAT fx = 0.0f;
    FLOAT fy = 0.0f;
    FLOAT fz = 0.0f;

    Particle pi = particles[i];

    for (int j = 0; j < n; j++)
    {
        Particle pj = particles[j];

        FLOAT dx = pj.x - pi.x;
        FLOAT dy = pj.y - pi.y;
        FLOAT dz = pj.z - pi.z;

        FLOAT distSqr = dx * dx + dy * dy + dz * dz + softening;
        FLOAT invDist = 1.0f / rsqrtf(distSqr);
        FLOAT invDist3 = invDist * invDist * invDist;

        FLOAT s = pj.mass * invDist3;

        fx += dx * s;
        fy += dy * s;
        fz += dz * s;
    }

    particles[i].vx += dt * fx;
    particles[i].vy += dt * fy;
    particles[i].vz += dt * fz;

    particles[i].x += dt * particles[i].vx;
    particles[i].y += dt * particles[i].vy;
    particles[i].z += dt * particles[i].vz;
}
