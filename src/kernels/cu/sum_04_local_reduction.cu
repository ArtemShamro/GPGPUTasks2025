#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "math.h"
#include <stdlib.h>

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];
    
    if (index >= n) {
        local_data[threadIdx.x] = 0;
    } else {
        local_data[threadIdx.x] = a[index];
    }
    __syncthreads();
    
    uint active_threads = blockDim.x / 2;
    for (; active_threads > 0; ) {
        if (threadIdx.x < active_threads) {
            local_data[threadIdx.x] += local_data[active_threads + threadIdx.x];
        }
        active_threads /= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        b[blockIdx.x] = local_data[threadIdx.x];
    }
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, gpu::gpu_mem_32u &sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    uint current_size = n;
    const gpu::gpu_mem_32u* input = &a; gpu::gpu_mem_32u* output = &sum;
    uint num_blocks;
    while (current_size > 1) {
        num_blocks = (current_size + GROUP_SIZE - 1) / GROUP_SIZE;
        ::sum_04_local_reduction<<<num_blocks, workSize.cuBlockSize(), 0, stream>>>(input->cuptr(), output->cuptr(), current_size);
        CUDA_CHECK_KERNEL(stream);
        input = &sum;
        current_size = num_blocks;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        sum_accum.cuptr(),      // куда копировать (результат)
        input->cuptr(),         // откуда (sum[0])
        sizeof(unsigned int),   // сколько байт
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}
} // namespace cuda
