#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static GLFWwindow *g_window = NULL;
static int g_width = 320, g_height = 240;

int lin_create_window(int width, int height) {
    if (!glfwInit()) {
        fprintf(stderr, "GLFW: Failed to initialize\n");
        return -1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    g_width = width;
    g_height = height;
    g_window = glfwCreateWindow(width, height, "Lin Window", NULL, NULL);
    if (!g_window) {
        fprintf(stderr, "GLFW: Failed to create window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(g_window);
    return 0;
}

int lin_window_should_close(void) {
    if (!g_window) return 1;
    return glfwWindowShouldClose(g_window) ? 1 : 0;
}

int lin_poll_events(void) {
    if (!g_window) return -1;
    glfwPollEvents();
    return 0;
}

int lin_display_pixels(int w, int h, int64_t data_ptr) {
    if (!g_window) return -1;
    unsigned char *pixels = (unsigned char*)(intptr_t)data_ptr;
    glfwMakeContextCurrent(g_window);
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(-1, 1);
    glPixelZoom(1.0f, -1.0f);
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glfwSwapBuffers(g_window);
    return 0;
}

int lin_destroy_window(void) {
    if (g_window) {
        glfwDestroyWindow(g_window);
        g_window = NULL;
    }
    glfwTerminate();
    return 0;
}