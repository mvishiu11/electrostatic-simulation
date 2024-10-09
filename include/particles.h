#ifndef PARTICLES_H
#define PARTICLES_H

// Particle data in Structure of Arrays (SoA) format
struct Particle {
    float *pos_x, *pos_y;     // Position
    float *vel_x, *vel_y;     // Velocity
    float *charge;            // Charge (positive for protons, negative for electrons)
};

// Initialize particles with random positions and velocities
void init_particles(Particle *particles, int num_particles);

// Update particles based on forces and update positions
__global__ void update_particles(float *pos_x, 
                                 float *pos_y, 
                                 float *vel_x, 
                                 float *vel_y, 
                                 float *charge, 
                                 int num_particles, 
                                 float dt,
                                 float boxWidth, 
                                 float boxHeight);

#endif // PARTICLES_H
