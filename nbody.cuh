#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

static __device__ inline float fast_rsqrtf(float a)
{
#if USE_NATIVE
    float r;
    asm("rsqrt.approx.ftz.f32 %0,%1; \n\t" : "=f"(r) : "f"(a));
    return r;
#else
    float r = __int_as_float(0x5f37642f - (__float_as_int(a) >> 1));
    r = fmaf(0.5f * r, fmaf(a * r, -r, 1.0f), r);
    float e = fmaf(a * r, -r, 1.0f);
    r = fmaf(fmaf(0.375f, e, 0.5f), e * r, r);
    return r;
#endif
}

__global__ void nbody_naive_kernel(Particle* particles, int n, float dt, float softening);
__global__ void nbody_tiled_kernel(Particle* particles, int n, float dt, float softening);

#endif // NBODY_CUH
