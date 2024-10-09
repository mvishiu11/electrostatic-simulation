#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call)                                 \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA Error: %s (error code %d)\n", \
                    cudaGetErrorString(err), (int)err);        \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#endif // CUDA_UTILS_H
