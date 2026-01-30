#include "nbody.cuh"
#include <cmath>

__global__ void nbody_naive_kernel(Particle *particles, int n, float dt, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    Particle pi = particles[i];

    for (int j = 0; j < n; j++)
    {
        Particle pj = particles[j];

        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;

        float distSqr = dx * dx + dy * dy + dz * dz + softening;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        float s = pj.mass * invDist3;

        fx += dx * s;
        fy += dy * s;
        fz += dz * s;
    }

    // G is assumed to be 1.0 for simplicity in this academic context
    // Unless specified otherwise.
    particles[i].vx += dt * fx;
    particles[i].vy += dt * fy;
    particles[i].vz += dt * fz;

    particles[i].x += dt * particles[i].vx;
    particles[i].y += dt * particles[i].vy;
    particles[i].z += dt * particles[i].vz;
}
