#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "particles.h"
#include "cuda_utils.h"

#define DOMAIN_SIZE 1.0f   // Simulation domain size (normalized to 1x1)

__device__ float compute_force(float charge1, float charge2, float distance) {
    const float ke = 8.99e9f; // Coulomb constant
    return ke * charge1 * charge2 / (distance * distance);
}

// Kernel to initialize particle positions and velocities
__global__ void init_particles_kernel(float *pos_x, float *pos_y, float *vel_x, float *vel_y, float *charge, int num_particles, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Random positions in the range [0, DOMAIN_SIZE]
        pos_x[idx] = curand_uniform(&state) * DOMAIN_SIZE;
        pos_y[idx] = curand_uniform(&state) * DOMAIN_SIZE;

        // Random velocities in the range [-0.01, 0.01]
        vel_x[idx] = (curand_uniform(&state) - 0.5f) * 0.02f;
        vel_y[idx] = (curand_uniform(&state) - 0.5f) * 0.02f;

        // Random charges, +1 for protons and -1 for electrons
        charge[idx] = (curand_uniform(&state) > 0.5f) ? 1.0f : -1.0f;
    }
}

// Host function to initialize particles
void init_particles(Particle *particles, int num_particles) {
    // Allocate memory on GPU
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles->pos_x, num_particles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles->pos_y, num_particles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles->vel_x, num_particles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles->vel_y, num_particles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles->charge, num_particles * sizeof(float)));

    // Launch kernel to initialize particles
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;
    init_particles_kernel<<<numBlocks, blockSize>>>(particles->pos_x, particles->pos_y, particles->vel_x, particles->vel_y, particles->charge, num_particles, time(NULL));

    // Synchronize to ensure initialization is complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void update_particles(float *pos_x, 
                                 float *pos_y, 
                                 float *vel_x, 
                                 float *vel_y, 
                                 float *charge, 
                                 int num_particles, 
                                 float dt, 
                                 float boxWidth, 
                                 float boxHeight) {
                                    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_particles) {
        float fx = 0.0f, fy = 0.0f;

        // Compute electrostatic force from other particles
        for (int j = 0; j < num_particles; ++j) {
            if (idx != j) {
                float dx = pos_x[j] - pos_x[idx];
                float dy = pos_y[j] - pos_y[idx];
                float dist = sqrtf(dx * dx + dy * dy + 0.01f); // Avoid division by zero

                // Compute force between particles
                float force = charge[idx] * charge[j] / (dist * dist + 1e-4f); // Coulomb force
                fx += force * dx / dist;
                fy += force * dy / dist;
            }
        }

        // Update velocity
        vel_x[idx] += fx * dt;
        vel_y[idx] += fy * dt;

        // Update position
        pos_x[idx] += vel_x[idx] * dt;
        pos_y[idx] += vel_y[idx] * dt;

        // Compute boundary limits
        float x_min = (1.0f - boxWidth) / 2.0f;
        float x_max = x_min + boxWidth;
        float y_min = (1.0f - boxHeight) / 2.0f;
        float y_max = y_min + boxHeight;

        // Apply bouncing off box boundaries
        if (pos_x[idx] < x_min || pos_x[idx] > x_max) {
            vel_x[idx] *= -1.0f;
            pos_x[idx] = fmaxf(x_min, fminf(pos_x[idx], x_max)); // Keep particle within bounds
        }
        if (pos_y[idx] < y_min || pos_y[idx] > y_max) {
            vel_y[idx] *= -1.0f;
            pos_y[idx] = fmaxf(y_min, fminf(pos_y[idx], y_max)); // Keep particle within bounds
        }
    }
}