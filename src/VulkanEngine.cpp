#include "VulkanEngine.hpp"
#include "EngineLog.hpp"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"
#include "ImGuizmo.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>

// ---------------------------------------------------------------------------
// checkLaunchConfig — detect exported-game mode via launch.cfg next to binary
// ---------------------------------------------------------------------------

void VulkanEngine::checkLaunchConfig() {
    std::ifstream f("launch.cfg");
    if (!f.is_open()) return;
    nlohmann::json cfg;
    try { f >> cfg; } catch (...) { return; }
    runtimeMode        = true;
    runtimeProjectPath = cfg.value("projectPath", "project");
    runtimeSceneName   = cfg.value("scene",       "main.scene");
    LOG_INFO("Runtime mode: project='%s' scene='%s'",
             runtimeProjectPath.c_str(), runtimeSceneName.c_str());
}

// ---------------------------------------------------------------------------
// Window creation (lives here because GLFWwindow* belongs to VulkanEngine)
// ---------------------------------------------------------------------------

static void glfwErrorCallback(int error, const char* description) {
    LOG_ERROR("GLFW error %d: %s", error, description);
}

void VulkanEngine::initWindow() {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        THROW_ENGINE_ERROR("Eroare fatala: GLFW nu s-a putut initializa!");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "3D RPG Maker - Depth Enabled", nullptr, nullptr);
    if (!window) {
        const char* description;
        glfwGetError(&description);
        THROW_ENGINE_ERROR(std::string("Eroare: Fereastra nu a putut fi creata! Cauza GLFW: ") +
                           (description ? description : "Necunoscuta"));
    }
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
}

// ---------------------------------------------------------------------------
// run() — top-level orchestration
// ---------------------------------------------------------------------------

void VulkanEngine::run() {
    checkLaunchConfig();
    initWindow();

    if (!runtimeMode) {
        GameObject testObj;
        testObj.id = activeScene.nextEntityId++;
        testObj.name = "Cub_Initial";
        testObj.transform.translation = glm::vec3(0.0f, 0.0f, 0.0f);
        testObj.transform.scale       = glm::vec3(1.0f, 1.0f, 1.0f);
        testObj.transform.rotation    = glm::vec3(0.0f, 0.0f, 0.0f);
        activeScene.entities.push_back(testObj);
    }

    initWorkspace();

    if (runtimeMode) {
        // Override workspace to point at the bundled project folder
        workspace.activeProject.name     = runtimeProjectPath;
        workspace.activeProject.rootPath = runtimeProjectPath;
        renderer.scanModelsFolder();
        renderer.scanTexturesFolder();
        renderer.scanAudioFolder();
    }

    renderer.init(window, &activeScene, &camera, &workspace);
    renderer.initImGui();
    runtime.init(&activeScene);

    EditorCallbacks cb;
    cb.refreshAssets       = [this]() { renderer.scanModelsFolder(); renderer.scanTexturesFolder(); renderer.scanAudioFolder(); };
    cb.applyModel          = [this]() { renderer.reloadCurrentModel(); };
    cb.saveScene           = [this](const std::string& f) { saveScene(f); };
    cb.loadScene           = [this](const std::string& f) { loadScene(f); };
    cb.refreshProjects     = [this]() { scanProjects(); };
    cb.switchProject       = [this](const std::string& n) { switchProject(n); };
    cb.createProject       = [this](const std::string& n) { createNewProject(n); };
    cb.isRayTracingReady   = [this]() { return renderer.rayTracer.isReady(); };
    cb.requestUndoSnapshot = [this]() { editor.pushUndo(); };
    cb.performUndo         = [this]() { editor.undo(); };
    cb.togglePlayMode      = [this]() { togglePlayMode(); };
    cb.isPlaying           = [this]() { return runtime.isPlaying(); };
    cb.onEntityDeleted     = [this](uint32_t id) {
        renderer.destroyMesh("__terrain_" + std::to_string(id));
        auto it = renderer.particleEmitters.find(id);
        if (it != renderer.particleEmitters.end()) {
            renderer.destroyEmitterState(it->second);
            renderer.particleEmitters.erase(it);
        }
        renderer.destroySkinInstance(id);
    };
    cb.getAnimationNames = [this](const std::string& modelName) -> std::vector<std::string> {
        std::vector<std::string> names;
        auto it = renderer.loadedSkins.find(modelName);
        if (it != renderer.loadedSkins.end())
            for (const auto& clip : it->second.animations)
                names.push_back(clip.name);
        return names;
    };
    cb.packAssets = [this]() { renderer.packAssets(); };
    cb.readScriptFile = [this](const std::string& rel) -> std::string {
        std::string path = workspace.activeProject.getScriptsPath() + "/" + rel;
        std::ifstream f(path);
        if (!f.is_open()) return "";
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    cb.writeScriptFile = [this](const std::string& rel, const std::string& content) {
        namespace fs = std::filesystem;
        fs::create_directories(workspace.activeProject.getScriptsPath());
        std::string path = workspace.activeProject.getScriptsPath() + "/" + rel;
        std::ofstream f(path);
        if (f.is_open()) f << content;
    };
    cb.listScriptFiles = [this]() -> std::vector<std::string> {
        namespace fs = std::filesystem;
        std::vector<std::string> out;
        std::string dir = workspace.activeProject.getScriptsPath();
        if (!fs::exists(dir)) return out;
        for (auto& entry : fs::directory_iterator(dir))
            if (entry.path().extension() == ".lua")
                out.push_back(entry.path().filename().string());
        return out;
    };
    cb.hotReloadScript = [this](uint32_t id, const std::string& code) {
        runtime.luaEngine.hotReloadScript(id, code);
    };
    cb.exportProject = [this](const std::string& startScene) {
        namespace fs = std::filesystem;
        std::string name    = workspace.activeProject.name;
        std::string distDir = "dist/" + name;
        std::error_code ec;

        fs::create_directories(distDir + "/shaders", ec);
        fs::create_directories(distDir + "/project", ec);

        // Copy this binary
        fs::path self = fs::read_symlink("/proc/self/exe", ec);
        if (!ec) {
            fs::copy(self, distDir + "/game",
                     fs::copy_options::overwrite_existing, ec);
            fs::permissions(distDir + "/game",
                fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                fs::perm_options::add, ec);
        }

        // Copy compiled shaders
        for (auto& e : fs::directory_iterator("shaders", ec))
            if (e.path().extension() == ".spv")
                fs::copy(e.path(), distDir + "/shaders/" + e.path().filename().string(),
                         fs::copy_options::overwrite_existing, ec);

        // Copy entire project folder
        fs::copy(workspace.activeProject.rootPath, distDir + "/project",
                 fs::copy_options::overwrite_existing | fs::copy_options::recursive, ec);

        // Write launch.cfg
        nlohmann::json cfg;
        cfg["projectPath"] = "project";
        cfg["scene"]       = startScene.empty() ? "main.scene" : startScene;
        std::ofstream cfgFile(distDir + "/launch.cfg");
        if (cfgFile.is_open()) cfgFile << cfg.dump(4) << '\n';

        // Write a shell launcher
        std::ofstream sh(distDir + "/run.sh");
        if (sh.is_open()) {
            sh << "#!/bin/bash\ncd \"$(dirname \"$0\")\"\n./game \"$@\"\n";
            fs::permissions(distDir + "/run.sh",
                fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                fs::perm_options::add, ec);
        }

        LOG_INFO("Exported to: %s", distDir.c_str());
    };

    runtime.cbIsKeyDown = [this](int key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };
    runtime.luaEngine.cbIsKeyDown = [this](int key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };
    runtime.luaEngine.cbPlaySound = [this](const std::string& filename) {
        runtime.audioEngine.playOneShot(workspace.activeProject.getAudioPath() + "/" + filename);
    };

    if (!runtimeMode) {
        editor.init(&activeScene, &workspace, &camera,
                    &renderer.availableModels, &renderer.availableTextures, &renderer.availableAudio,
                    &renderer.selectedModelIndex, &renderer.selectedTextureIndex,
                    &renderer.useRayTracing, &activeScene.shadowsEnabled, std::move(cb));
    }

    if (runtimeMode) {
        // Load the starting scene and immediately enter play mode
        loadScene(workspace.activeProject.getScenesPath() + "/" + runtimeSceneName);
        glfwSetWindowTitle(window, workspace.activeProject.name.c_str());
        togglePlayMode();
    }

    mainLoop();
    vkDeviceWaitIdle(renderer.device);
    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
}

// ---------------------------------------------------------------------------
// mainLoop
// ---------------------------------------------------------------------------

void VulkanEngine::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();
        processInput(window);

        bool playing = runtime.isPlaying();
        if (playing) {
            runtime.tick(deltaTime, camera);
            renderer.tickInGameUI(true);

            // Intercept loadScene requests and run a fade transition instead
            if (!runtime.luaEngine.pendingSceneLoad.empty() && !fadeActive) {
                pendingTransitionScene = runtime.luaEngine.pendingSceneLoad;
                runtime.luaEngine.pendingSceneLoad = "";
                fadeActive = true;
                fadingOut  = true;
                fadeAlpha  = 0.0f;
            }
        }

        // Tick fade (runs even while scene is loading so fade-in works after reload)
        if (fadeActive) {
            constexpr float kFadeSpeed = 2.5f; // 0-1 in 0.4 s
            if (fadingOut) {
                fadeAlpha += kFadeSpeed * deltaTime;
                if (fadeAlpha >= 1.0f) {
                    fadeAlpha = 1.0f;
                    if (!pendingTransitionScene.empty()) {
                        loadScene(pendingTransitionScene);
                        pendingTransitionScene.clear();
                    }
                    fadingOut = false;
                }
            } else {
                fadeAlpha -= kFadeSpeed * deltaTime;
                if (fadeAlpha <= 0.0f) {
                    fadeAlpha  = 0.0f;
                    fadeActive = false;
                }
            }
        }

        renderUI();
        renderer.drawFrame(deltaTime, playing);
    }
}

// ---------------------------------------------------------------------------
// renderUI — builds ImGui frame; Renderer::drawFrame consumes GetDrawData()
// ---------------------------------------------------------------------------

void VulkanEngine::renderUI() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    if (!runtimeMode) editor.render();

    if (runtime.isPlaying()) renderDialogue();

    if (fadeActive && fadeAlpha > 0.0f) {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        ImVec2 sz = ImGui::GetIO().DisplaySize;
        dl->AddRectFilled({0, 0}, sz, IM_COL32(0, 0, 0, (int)(fadeAlpha * 255)));
    }

    if (uiCallback) uiCallback();
    ImGui::Render();
}

// ---------------------------------------------------------------------------
// togglePlayMode
// ---------------------------------------------------------------------------

void VulkanEngine::togglePlayMode() {
    if (!runtime.isPlaying()) {
        renderer.uiPrevMouseDown = false;
        runtime.start(camera,
                      workspace.activeProject.getScriptsPath(),
                      workspace.activeProject.getAudioPath(),
                      renderer.buildNavmesh());
    } else {
        renderer.uiPrevMouseDown = false;
        runtime.stop(camera);
    }
}

// ---------------------------------------------------------------------------
// renderDialogue — ImGui dialogue panel (play mode only)
// ---------------------------------------------------------------------------

void VulkanEngine::renderDialogue() {
    if (!runtime.luaEngine.dialogue.active) return;
    const auto& d = runtime.luaEngine.dialogue;

    ImGuiIO& io = ImGui::GetIO();
    float w = io.DisplaySize.x;
    float h = io.DisplaySize.y;
    float panelH = d.choices.empty() ? 150.0f : 150.0f + d.choices.size() * 32.0f;

    ImGui::SetNextWindowPos({w * 0.05f, h - panelH - 20.0f});
    ImGui::SetNextWindowSize({w * 0.9f, panelH});
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::Begin("##dialogue", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs * d.choices.empty() |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);

    ImGui::TextColored({1.0f, 0.85f, 0.2f, 1.0f}, "%s", d.speaker.c_str());
    ImGui::Separator();
    ImGui::TextWrapped("%s", d.text.c_str());

    if (d.choices.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("[Press ENTER to continue]");
        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_Space))
            runtime.luaEngine.dialogue.dismissed = true;
    } else {
        ImGui::Separator();
        for (int i = 0; i < (int)d.choices.size(); i++) {
            std::string label = std::to_string(i + 1) + ". " + d.choices[i];
            if (ImGui::Button(label.c_str(), {-1, 0}))
                runtime.luaEngine.dialogue.selectedChoice = i;
        }
    }

    ImGui::End();
}

// ---------------------------------------------------------------------------
// cleanup (called after vkDeviceWaitIdle in run())
// ---------------------------------------------------------------------------

void VulkanEngine::cleanup() {
    // renderer.cleanup() and glfwDestroyWindow are called inline in run()
    // This method is kept for any future VulkanEngine-level teardown.
}
