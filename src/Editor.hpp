#pragma once
#include "Scene.hpp"
#include "Project.hpp"
#include "Camera.hpp"
#include "EditorUI.hpp"
#include <vector>
#include <set>
#include <string>

// ---------------------------------------------------------------------------
// Editor — owns all editor-only state: selection, clipboard, undo/redo,
// scene-dirty flag, and the EditorUI widget.
// Operates on a Scene* provided by VulkanEngine (activeScene stays there).
// ---------------------------------------------------------------------------

class Editor {
public:
    void init(Scene* scene, Workspace* workspace, Camera* camera,
              std::vector<std::string>* models,
              std::vector<std::string>* textures,
              std::vector<std::string>* audio,
              int* selectedModel, int* selectedTexture,
              bool* useRayTracing, bool* shadowsEnabled,
              EditorCallbacks cb);

    // Call between ImGui::NewFrame() / ImGui::Render() each frame.
    void render();

    void pushUndo();
    void undo();
    void redo();

    // Public — accessed directly by VulkanInput and VulkanAssets.
    int                     selectedEntityIndex = -1;
    std::set<int>           selectedEntities;
    std::vector<GameObject> clipboard;
    bool                    hasUnsavedChanges   = false;
    std::string             currentSceneFile    = "nivel_test.scene";

private:
    Scene*   scene = nullptr;
    EditorUI editorUI;

    static constexpr int kMaxUndoHistory = 50;
    std::vector<Scene> undoStack;
    std::vector<Scene> redoStack;
};
