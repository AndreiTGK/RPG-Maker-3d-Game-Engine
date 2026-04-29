#pragma once
#include "Scene.hpp"
#include "Project.hpp"
#include "Camera.hpp"
#include <vector>
#include <string>
#include <functional>
#include <set>

struct EditorCallbacks {
    std::function<void()>                   refreshAssets;      // rescan models + textures
    std::function<void()>                   applyModel;         // vkWaitIdle + load mesh into cache
    std::function<void(const std::string&)> saveScene;          // filename includes .scene
    std::function<void(const std::string&)> loadScene;          // filename includes .scene
    std::function<void()>                   refreshProjects;
    std::function<void(const std::string&)> switchProject;
    std::function<void(const std::string&)> createProject;
    std::function<bool()>                   isRayTracingReady;
    std::function<void()>                   requestUndoSnapshot; // call BEFORE making a change
    std::function<void()>                   performUndo;         // pop last snapshot
    std::function<void()>                   togglePlayMode;
    std::function<bool()>                   isPlaying;
    std::function<void(uint32_t entityId)>  onEntityDeleted; // clean up GPU resources
    // Returns animation clip names for a given skinned mesh model (empty if not loaded)
    std::function<std::vector<std::string>(const std::string&)> getAnimationNames;
    // Pack all project assets into a .rpak file
    std::function<void()> packAssets;
    // Export project to dist/<name>/ as a standalone runnable bundle
    std::function<void(const std::string& startScene)> exportProject;
    // Script file I/O (relative paths within project scripts/ dir)
    std::function<std::string(const std::string&)>              readScriptFile;
    std::function<void(const std::string&, const std::string&)> writeScriptFile;
    std::function<std::vector<std::string>()>                   listScriptFiles;
    // Hot-reload: swap entity's Lua state mid-play from a code string
    std::function<void(uint32_t, const std::string&)>           hotReloadScript;
};

class EditorUI {
public:
    // Call once after ImGui context is created — sets the editor colour theme
    void applyTheme();

    void init(Scene* scene, Workspace* workspace, Camera* camera,
              std::vector<std::string>* models, std::vector<std::string>* textures,
              std::vector<std::string>* audio,
              int* selectedModel, int* selectedTexture,
              bool* useRayTracing, bool* shadowsEnabled, EditorCallbacks callbacks);

    // Called between ImGui::NewFrame() / ImGui::Render() each frame
    void render(int& selectedEntityIndex, std::set<int>& selectedEntities,
                bool& hasUnsavedChanges, std::string& currentSceneFile);

private:
    Scene*                    scene          = nullptr;
    Workspace*                workspace      = nullptr;
    Camera*                   camera         = nullptr;
    std::vector<std::string>* models         = nullptr;
    std::vector<std::string>* textures       = nullptr;
    std::vector<std::string>* audio          = nullptr;
    int*                      selectedModel  = nullptr;
    int*                      selectedTexture= nullptr;
    bool*                     useRayTracing   = nullptr;
    bool*                     shadowsEnabled  = nullptr;
    EditorCallbacks           cb;

    // UI-local state
    bool        showOverwriteModal   = false;
    bool        showUnsavedModal     = false;
    std::string pendingSceneToLoad;
    char        newProjectNameBuffer[256] = {};
    int         selectedProjectIndex = 0;

    // Snapping
    bool  snapEnabled    = false;
    float snapTranslate  = 0.5f;   // world units
    float snapRotate     = 15.0f;  // degrees
    float snapScale      = 0.1f;   // scale increment

    // Script editor
    bool        showScriptEditor     = false;
    std::string scriptEditorContent;          // text being edited (heap)
    std::string scriptEditorFile;             // relative filename currently open
    bool        scriptEditorDirty    = false;
    uint32_t    scriptEditorEntityId = 0;     // entity that owns the open script

    // Visual script editor (Scratch-like window)
    bool        showVisualEditor     = false;
    uint32_t    visualEditorEntityId = 0;
    int         vsActiveStack        = 0;     // which stack palette clicks append to

    // Prefab
    char        prefabNameBuf[128]   = {};
    int         selectedPrefabIdx    = -1;
    uint32_t    pendingSavePrefabId  = 0;     // entity id to save (set from context menu)

    void renderVisualEditorWindow();
};
