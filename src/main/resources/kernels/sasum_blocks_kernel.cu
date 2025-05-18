extern "C"
__global__ void sasum_blocks_kernel(
    const float* __restrict__ hist,  // hossz n
    const float* __restrict__ act,   // hossz m*n
    int n,
    float* sums)                     // hossz m
{
    extern __shared__ float sdata[];
    int tid   = threadIdx.x;
    int bid   = blockIdx.x;          // bid ∈ [0, m)
    if (bid >= gridDim.x) return;

    // pointer az aktuális act-blokkra
    const float* act_block = act + (size_t)bid * n;

    // thread‐szintű részösszeg
    float acc = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        acc += fabsf(hist[j] - act_block[j]);
    }
    sdata[tid] = acc;
    __syncthreads();

    // redukció shared mem-ben
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // írd ki az eredményt sums[bid]-be
    if (tid == 0) {
        sums[bid] = sdata[0];
    }
}
