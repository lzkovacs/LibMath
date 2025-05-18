extern "C"
__global__ void fillKernel(float* d_vec, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_vec[idx] = value;
    }
}
