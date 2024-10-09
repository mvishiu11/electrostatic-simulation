#include <cuda_runtime.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>
#include "cuda_utils.h"
#include "particles.h"
#include "visualization.h"
#include <stdio.h>

// Simulation parameters
const int num_particles = 10000;        // Number of particles
const int width = 800;                  // Window width
const int height = 600;                 // Window height
float timeScale = 1.0f;                 // Multiplier for the simulation speed
float dt = 0.01f;                       // Define a base time step for particle updates
bool paused = false;                    // Flag to pause/unpause the simulation
float intensityScale = 1.0f;            // Scale factor for field intensity
Particle particles;                    // Particle data structure
float *field;                           // Field intensity buffer
GLuint fieldTexture;                    // OpenGL texture for field visualization
float *field_rgb;                       // Buffer for field color data (RGB)
bool showField = true;                  // Flag to toggle field visualization

// Box boundary parameters (3/4 of the screen dimensions)
const float boxWidth = 0.75f;
const float boxHeight = 0.75f;

// Initialize CUDA resources
void init_cuda_resources() {
    CHECK_CUDA_ERROR(cudaMallocManaged(&field, width * height * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&field_rgb, width * height * 3 * sizeof(float)));  // RGB color buffer
}

// Initialize OpenGL
void initOpenGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA OpenGL Electrostatic Field Visualization");

    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(glew_status));
        exit(EXIT_FAILURE);
    }

    // Set up the texture for field visualization
    glGenTextures(1, &fieldTexture);
    glBindTexture(GL_TEXTURE_2D, fieldTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

// Keyboard controls for pausing, speed, and intensity adjustment
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 'p':
            paused = !paused;           // Pause/unpause
            break;
        case '+':
            timeScale *= 1.1f;          // Speed up
            break;
        case '-':
            timeScale /= 1.1f;          // Slow down
            break;
        case 'i':
            intensityScale *= 1.1f;     // Increase intensity
            break;
        case 'd':
            intensityScale /= 1.1f;     // Decrease intensity
            break;
        case 'f':                       // Toggle field visualization
            showField = !showField;
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

// OpenGL display function
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    if (!paused) {
        // Update particle positions with time scaling
        int blockSize = 256;
        int numBlocks = (num_particles + blockSize - 1) / blockSize;
        update_particles<<<numBlocks, blockSize>>>(particles.pos_x, 
                                                   particles.pos_y, 
                                                   particles.vel_x, 
                                                   particles.vel_y, 
                                                   particles.charge, 
                                                   num_particles, 
                                                   dt * timeScale,
                                                   boxWidth,
                                                   boxHeight);
        cudaDeviceSynchronize();

        // Compute the electrostatic field
        compute_field(field, &particles, num_particles, width, height);
    }

    // Bind field data to the texture
    glBindTexture(GL_TEXTURE_2D, fieldTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, field);

    // Render the field texture as a background, only if `showField` is true
    if (showField) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float intensity = field[idx] * intensityScale;
    
                int rgb_idx = idx * 3;
                if (intensity > 0) {
                    // Positive values to red
                    field_rgb[rgb_idx] = intensity;
                    field_rgb[rgb_idx + 1] = 0.0f;
                    field_rgb[rgb_idx + 2] = 0.0f;
                } else {
                    // Negative values to blue
                    field_rgb[rgb_idx] = 0.0f;
                    field_rgb[rgb_idx + 1] = 0.0f;
                    field_rgb[rgb_idx + 2] = -intensity;
                }
            }
        }
        
        // Bind the texture with RGB data
        glBindTexture(GL_TEXTURE_2D, fieldTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, field_rgb);
    
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);
    }    

    // Draw a bounding box
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINE_LOOP);
        float x_min = (1.0f - boxWidth) / 2.0f * 2.0f - 1.0f;
        float x_max = x_min + boxWidth * 2.0f;
        float y_min = (1.0f - boxHeight) / 2.0f * 2.0f - 1.0f;
        float y_max = y_min + boxHeight * 2.0f;

        glVertex2f(x_min, y_min);
        glVertex2f(x_max, y_min);
        glVertex2f(x_max, y_max);
        glVertex2f(x_min, y_max);
    glEnd();

    // Render particles as points with different colors for protons and electrons
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < num_particles; i++) {
        float x = particles.pos_x[i] * 2.0f - 1.0f;
        float y = particles.pos_y[i] * 2.0f - 1.0f;

        // Color particles based on charge
        if (particles.charge[i] > 0) {
            glColor3f(1.0f, 0.0f, 0.0f);
            glPointSize(7.0f);          
        } else {
            glColor3f(0.0f, 0.0f, 1.0f);
            glPointSize(5.0f);           
        }

        glVertex2f(x, y);
    }
    glEnd();

    // Swap buffers to display the updated scene
    glutSwapBuffers();
}

int main(int argc, char **argv) {
    // Initialize OpenGL
    initOpenGL(&argc, argv);

    // Initialize CUDA resources
    init_cuda_resources();

    // Initialize particles
    init_particles(&particles, num_particles);

    // Register OpenGL callbacks
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutKeyboardFunc(keyboard);

    // Main loop
    glutMainLoop();

    // Free resources on exit
    cudaFree(particles.pos_x);
    cudaFree(particles.pos_y);
    cudaFree(particles.vel_x);
    cudaFree(particles.vel_y);
    cudaFree(particles.charge);
    cudaFree(field);
    cudaFree(field_rgb);

    return 0;
}