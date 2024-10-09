#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include "opengl_utils.h"

void initOpenGL(int *argc, char **argv) {
    // Initialize GLUT
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(800, 800);  // Default window size
    glutCreateWindow("Electrostatic Field Simulation");

    // Initialize GLEW
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(glew_status));
        exit(EXIT_FAILURE);
    }
    
    // Set up some default OpenGL settings
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Background color (black)
}
