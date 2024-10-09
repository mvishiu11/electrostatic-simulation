#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

void init() {
    glewInit();
    if (glewIsSupported("GL_VERSION_4_5")) {
        std::cout << "OpenGL 4.5 is supported" << std::endl;
    } else {
        std::cout << "OpenGL 4.5 not supported" << std::endl;
    }

    if (glewIsSupported("GL_NV_vertex_buffer_unified_memory") ||
        glewIsSupported("GL_ARB_vertex_buffer_object")) {
        std::cout << "CUDA-OpenGL interop extensions are available" << std::endl;
    } else {
        std::cout << "CUDA-OpenGL interop extensions are not available" << std::endl;
    }
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutCreateWindow("OpenGL Test");
    init();
    return 0;
}