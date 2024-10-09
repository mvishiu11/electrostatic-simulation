#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "particles.h"
#include "visualization.h"

#define DOMAIN_SIZE 1.0f

__device__ float compute_field(float charge, float dist) {
    // Compute field intensity using an inverse-square law
    return charge / (dist * dist + 0.01f);  // Adding 0.01 to avoid division by zero
}

__global__ void compute_field_kernel(float *field, float *pos_x, float *pos_y, float *charge, int num_particles, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        float px = (float)x / width * DOMAIN_SIZE;
        float py = (float)y / height * DOMAIN_SIZE;
        float intensity = 0.0f;

        // Accumulate field intensity from each particle
        for (int i = 0; i < num_particles; ++i) {
            float dx = pos_x[i] - px;
            float dy = pos_y[i] - py;
            float dist = sqrtf(dx * dx + dy * dy + 0.01f);
            intensity += compute_field(charge[i], dist);
        }

        // Store the calculated intensity in the buffer
        field[idx] = intensity;
    }
}

// Host function to call the field kernel
void compute_field(float *field, Particle *particles, int num_particles, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    compute_field_kernel<<<numBlocks, blockSize>>>(field, particles->pos_x, particles->pos_y, particles->charge, num_particles, width, height);
    cudaDeviceSynchronize();
}
