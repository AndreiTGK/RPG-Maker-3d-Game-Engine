#include "VulkanEngine.hpp"
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "ImGuizmo.h"

void VulkanEngine::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    app->renderer.framebufferResized = true;
}

void VulkanEngine::processInput(GLFWwindow* window) {
    // Editor-only shortcuts — disabled in play mode and runtime (exported game) mode
    if (!runtimeMode && !ImGui::GetIO().WantCaptureKeyboard && !runtime.isPlaying()) {
        bool ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        static bool wasZ = false, wasY = false;
        bool isZ = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
        bool isY = glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS;

        if (ctrl && isZ && !wasZ) editor.undo();
        if (ctrl && isY && !wasY) editor.redo();

        wasZ = isZ;
        wasY = isY;

        // Copy — Ctrl+C: copy selected entities to clipboard
        static bool wasC = false, wasV = false, wasD = false;
        bool isC = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
        bool isV = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
        bool isD = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

        if (ctrl && isC && !wasC) {
            editor.clipboard.clear();
            std::set<int> toClip = editor.selectedEntities;
            if (toClip.empty() && editor.selectedEntityIndex >= 0) toClip.insert(editor.selectedEntityIndex);
            for (int idx : toClip)
                if (idx >= 0 && idx < (int)activeScene.entities.size())
                    editor.clipboard.push_back(activeScene.entities[idx]);
        }

        // Paste — Ctrl+V: paste clipboard entities into scene
        if (ctrl && isV && !wasV && !editor.clipboard.empty()) {
            editor.pushUndo();
            editor.selectedEntities.clear();
            int firstNew = (int)activeScene.entities.size();
            for (auto obj : editor.clipboard) {
                obj.id = activeScene.nextEntityId++;
                obj.parentId = 0;
                obj.transform.translation += glm::vec3(0.5f, 0.5f, 0.0f);
                activeScene.entities.push_back(obj);
            }
            editor.selectedEntityIndex = firstNew;
            for (int i = firstNew; i < (int)activeScene.entities.size(); i++)
                editor.selectedEntities.insert(i);
            editor.hasUnsavedChanges = true;
        }

        // Duplicate — Ctrl+D: copy + paste in one step
        if (ctrl && isD && !wasD) {
            editor.pushUndo();
            std::set<int> toDup = editor.selectedEntities;
            if (toDup.empty() && editor.selectedEntityIndex >= 0) toDup.insert(editor.selectedEntityIndex);
            editor.selectedEntities.clear();
            int firstNew = (int)activeScene.entities.size();
            for (int idx : toDup) {
                if (idx < 0 || idx >= (int)activeScene.entities.size()) continue;
                GameObject obj = activeScene.entities[idx];
                obj.id = activeScene.nextEntityId++;
                obj.parentId = 0;
                obj.transform.translation += glm::vec3(0.5f, 0.5f, 0.0f);
                activeScene.entities.push_back(obj);
            }
            if (!toDup.empty()) {
                editor.selectedEntityIndex = firstNew;
                for (int i = firstNew; i < (int)activeScene.entities.size(); i++)
                    editor.selectedEntities.insert(i);
            }
            editor.hasUnsavedChanges = true;
        }

        wasC = isC; wasV = isV; wasD = isD;
    }

    // WASD/EQ camera movement: editor only. In play mode, tickPlayerController handles WASD.
    if (!runtimeMode && !runtime.isPlaying()) {
        if (camera.orthoMode) {
            // Ortho/2D mode: WASD pans on the XY plane; zoom via scroll in EditorUI
            float panSpeed = camera.orthoZoom * 1.5f * deltaTime;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.pos.y += panSpeed;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.pos.y -= panSpeed;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.pos.x -= panSpeed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.pos.x += panSpeed;
        } else {
            float cameraSpeed = 2.5f * deltaTime;

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                camera.pos += cameraSpeed * camera.front;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                camera.pos -= cameraSpeed * camera.front;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                camera.pos -= glm::normalize(glm::cross(camera.front, camera.up)) * cameraSpeed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                camera.pos += glm::normalize(glm::cross(camera.front, camera.up)) * cameraSpeed;

            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                camera.pos += cameraSpeed * camera.up;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                camera.pos -= cameraSpeed * camera.up;
        }
    }
    static bool wasLeftMousePressed = false;
    bool isLeftMousePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    if (!runtime.isPlaying() && isLeftMousePressed && !wasLeftMousePressed) {
        if (!ImGui::GetIO().WantCaptureMouse && !ImGuizmo::IsOver()) {
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            glm::vec3 rayDir = renderer.getRayFromMouse(mouseX, mouseY);

            int hitIndex = -1;
            float closestDist = 99999.0f;

            for (int i = 0; i < (int)activeScene.entities.size(); i++) {
                float dist;
                glm::mat4 world = getWorldTransform(activeScene, i);
                glm::vec3 worldPos = glm::vec3(world[3]);
                glm::vec3 worldScale = {
                    glm::length(glm::vec3(world[0])),
                    glm::length(glm::vec3(world[1])),
                    glm::length(glm::vec3(world[2]))
                };
                float radius = glm::max(worldScale.x, glm::max(worldScale.y, worldScale.z));

                if (renderer.raySphereIntersect(camera.pos, rayDir, worldPos, radius, dist)) {
                    if (dist < closestDist) {
                        closestDist = dist;
                        hitIndex = i;
                    }
                }
            }
            bool shiftHeld = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                             glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

            if (hitIndex >= 0 && shiftHeld) {
                if (editor.selectedEntities.count(hitIndex))
                    editor.selectedEntities.erase(hitIndex);
                else
                    editor.selectedEntities.insert(hitIndex);
                editor.selectedEntityIndex = hitIndex;
            } else {
                editor.selectedEntities.clear();
                editor.selectedEntityIndex = hitIndex;
            }
        }
    }
    wasLeftMousePressed = isLeftMousePressed;
}

void VulkanEngine::mouseCallback(GLFWwindow* window, double xposIn, double yposIn) {
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (!app->camera.orthoMode && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        if (app->camera.firstMouse) {
            app->camera.lastX = xpos;
            app->camera.lastY = ypos;
            app->camera.firstMouse = false;
        }

        float xoffset = app->camera.lastX - xpos;
        float yoffset = app->camera.lastY - ypos;

        app->camera.lastX = xpos;
        app->camera.lastY = ypos;

        float sensitivity = 0.2f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        app->camera.yaw   += xoffset;
        app->camera.pitch += yoffset;

        if (app->camera.pitch >  89.0f) app->camera.pitch =  89.0f;
        if (app->camera.pitch < -89.0f) app->camera.pitch = -89.0f;

        glm::vec3 direction;
        direction.x = cos(glm::radians(app->camera.yaw)) * cos(glm::radians(app->camera.pitch));
        direction.y = sin(glm::radians(app->camera.yaw)) * cos(glm::radians(app->camera.pitch));
        direction.z = sin(glm::radians(app->camera.pitch));

        app->camera.front = glm::normalize(direction);
    } else {
        app->camera.firstMouse = true;
    }
}
