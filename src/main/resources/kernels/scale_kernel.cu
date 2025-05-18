typedef float4 Vec4;
extern "C"
__global__ void scaleKernelVec4(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    float diff, float base, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = tid * 4;
    if (idx4 + 3 < N) {
        Vec4 v = reinterpret_cast<const Vec4*>(x)[tid];
        v.x = fmaf(v.x, diff, base);
        v.y = fmaf(v.y, diff, base);
        v.z = fmaf(v.z, diff, base);
        v.w = fmaf(v.w, diff, base);
        reinterpret_cast<Vec4*>(y)[tid] = v;
    } else {
        for (int i = idx4; i < N; ++i) {
            float u = x[i];
            y[i] = fmaf(u, diff, base);
        }
    }
}
