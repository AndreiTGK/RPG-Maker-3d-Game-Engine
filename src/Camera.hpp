#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera {
    glm::vec3 pos   = { 2.0f,  2.0f,  2.0f};
    glm::vec3 front = {-1.0f, -1.0f, -1.0f};
    glm::vec3 up    = { 0.0f,  0.0f,  1.0f};
    float yaw       = -90.0f;
    float pitch     = 0.0f;
    float lastX     = 400.0f;
    float lastY     = 300.0f;
    bool firstMouse = true;

    bool  orthoMode = false;
    float orthoZoom = 10.0f; // half-height in world units

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(pos, pos + front, up);
    }

    glm::mat4 getProjMatrix(float aspect) const {
        if (orthoMode) {
            float halfH = orthoZoom;
            float halfW = halfH * aspect;
            return glm::ortho(-halfW, halfW, -halfH, halfH, -1000.0f, 1000.0f);
        }
        return glm::perspective(glm::radians(45.0f), aspect, 0.1f, 1000.0f);
    }
};
