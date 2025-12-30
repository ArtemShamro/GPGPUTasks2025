#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include <stdlib.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// --- Вспомогательная warp-редукция без барьеров ---
__inline__ __device__ unsigned int warp_reduce_sum(unsigned int v) {
    // полная маска для всех активных потоков варпа
    unsigned mask = 0xFFFFFFFFu;
    // шаги 16,8,4,2,1
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// --- Быстрая редукция: многократные загрузки + warp-шаги + 1 atomic на блок ---
__global__ void sum_05_reduction_one_kernel_fast(
    const unsigned int* __restrict__ a,
    unsigned int* __restrict__ out,
    unsigned int n)
{
    const unsigned int tid    = threadIdx.x;
    const unsigned int lane   = tid & (WARP_SIZE - 1);         // номер потока внутри варпа
    const unsigned int warpId = tid >> 5;                      // номер варпа внутри блока
    const unsigned int blockThreads = blockDim.x;
    const unsigned int gridStride   = blockThreads * gridDim.x;

    // --- 1) Каждому потоку: суммировать много элементов (grid-stride + unroll) ---
    unsigned int sum = 0;

    // Небольшой ручной анролл на 4 прохода по сети грид-страйда
    // Это сильно экономит адресную арифметику и помогает скрыть латентность памяти
    unsigned int i = blockIdx.x * blockThreads + tid;
    for (; i < n; i += gridStride * 4) {
        unsigned int i1 = i;
        unsigned int i2 = i + gridStride;
        unsigned int i3 = i + gridStride * 2;
        unsigned int i4 = i + gridStride * 3;

        if (i1 < n) sum += a[i1];
        if (i2 < n) sum += a[i2];
        if (i3 < n) sum += a[i3];
        if (i4 < n) sum += a[i4];
    }

    // --- 2) Редукция внутри варпа (без синхронизаций блока) ---
    sum = warp_reduce_sum(sum);

    // --- 3) Сложим суммы варпов в shared и финализируем одной warp ---
    __shared__ unsigned int warpSums[GROUP_SIZE / WARP_SIZE]; // GROUP_SIZE равен blockDim.x
    if (lane == 0) {
        warpSums[warpId] = sum;   // один поток на варп пишет сумму
    }
    __syncthreads();

    // --- 4) Первая warp блока редуцирует warpSums ---
    if (warpId == 0) {
        unsigned int v = (tid < (blockThreads / WARP_SIZE)) ? warpSums[lane] : 0u;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            // Один atomic на блок — это дешево
            atomicAdd(out, v);
        }
    }
}

namespace cuda {

void sum_05_reduction_one_kernel(const gpu::WorkSize &workSize,
                                 const gpu::gpu_mem_32u &a,
                                 gpu::gpu_mem_32u &sum,
                                 unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();

    // Важно: у нас grid-stride loop, так что gridSize/blockSize из workSize подойдут любые “разумные”.
    // Но на практике быстрее всего: blockDim = 256 или 512; gridDim = 8–16 * SM_count.
    // Здесь уважаем пришедший workSize:
    const dim3 block(workSize.cuBlockSize());
    const dim3 grid (workSize.cuGridSize());

    ::sum_05_reduction_one_kernel_fast<<<grid, block, 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
