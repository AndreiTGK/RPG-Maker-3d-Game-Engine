#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "Camera.hpp"
#include "Scene.hpp"
#include "Project.hpp"
#include "Runtime.hpp"
#include "Editor.hpp"
#include "Renderer.hpp"
#include <string>
#include <functional>

class VulkanEngine {
public:
    void run();
    void setUIFunc(std::function<void()> func) { uiCallback = func; }
    std::function<void()> uiCallback;

private:
    GLFWwindow* window    = nullptr;
    Scene       activeScene;
    Camera      camera;
    Workspace   workspace;

    Runtime  runtime;
    Editor   editor;
    Renderer renderer;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    // Scene transition fade
    float       fadeAlpha  = 0.0f;
    bool        fadingOut  = false;
    bool        fadeActive = false;
    std::string pendingTransitionScene;

    // Runtime (exported game) mode — editor UI skipped
    bool        runtimeMode        = false;
    std::string runtimeProjectPath;  // relative path to project root
    std::string runtimeSceneName;    // scene filename (no path)

    // --- Core loop ---
    void initWindow();
    void mainLoop();
    void cleanup();

    // --- Play mode ---
    void togglePlayMode();

    // --- UI (ImGui) ---
    void renderUI();
    void renderDialogue();

    // --- Scene / project management ---
    void saveScene(const std::string& filename);
    void loadScene(const std::string& filename);
    void initWorkspace();
    void scanProjects();
    void switchProject(const std::string& projectName);
    void createNewProject(const std::string& projectName);
    void checkLaunchConfig();   // detect exported-game mode via launch.cfg

    // --- Input (VulkanInput.cpp) ---
    void processInput(GLFWwindow* window);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
};
