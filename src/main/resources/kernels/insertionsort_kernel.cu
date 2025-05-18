extern "C"
__global__ void sortKernel(float *data, int N, int j) {
    int start = (blockIdx.x * blockDim.x + threadIdx.x) * j;

    if (start < N) {
        // Sort `j` element (esetünkben j=visiblecolumnsize, azaz 5 lesz)
        for (int i = start; i < min(start + j, N); i++) {
            float key = data[i];
            int k = i - 1;
            while (k >= start && data[k] > key) {
                data[k + 1] = data[k];
                k--;
            }
            data[k + 1] = key;
        }
    }
}