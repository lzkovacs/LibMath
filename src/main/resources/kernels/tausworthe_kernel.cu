#include <curand_kernel.h>

extern "C"
__global__ void combinedTauswortheKernel(float* result, int N, int min, int max, long seed1, long seed2, long seed3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Define the Tausworthe parameters
    long s1 = seed1 + idx;
    long s2 = seed2 + idx;
    long s3 = seed3 + idx;

    int range = max - min + 1;

    // Tausworthe function
    auto tausworthe = [](long z, int s, int q, int r, long m) -> long {
        long b = (((z << q) ^ z) >> s);
        return ((z & m) << r) ^ b;
    };

    // Tausworthe step
    s1 = tausworthe(s1, 13, 19, 12, 4294967294L);
    s2 = tausworthe(s2, 2, 25, 4, 4294967288L);
    s3 = tausworthe(s3, 3, 11, 17, 4294967280L);
    long combined = s1 ^ s2 ^ s3;

    // Generate random float between min and max
    result[idx] = min + (float)((combined >> 1) % range);
}

