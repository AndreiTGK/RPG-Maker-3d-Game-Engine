#include "Editor.hpp"

void Editor::init(Scene* s, Workspace* workspace, Camera* camera,
                  std::vector<std::string>* models,
                  std::vector<std::string>* textures,
                  std::vector<std::string>* audio,
                  int* selectedModel, int* selectedTexture,
                  bool* useRayTracing, bool* shadowsEnabled,
                  EditorCallbacks cb) {
    scene = s;
    editorUI.init(s, workspace, camera,
                  models, textures, audio,
                  selectedModel, selectedTexture,
                  useRayTracing, shadowsEnabled,
                  std::move(cb));
}

void Editor::render() {
    editorUI.render(selectedEntityIndex, selectedEntities,
                    hasUnsavedChanges, currentSceneFile);
}

void Editor::pushUndo() {
    undoStack.push_back(*scene);
    if ((int)undoStack.size() > kMaxUndoHistory)
        undoStack.erase(undoStack.begin());
    redoStack.clear();
}

void Editor::undo() {
    if (undoStack.empty()) return;
    redoStack.push_back(*scene);
    *scene = undoStack.back();
    undoStack.pop_back();
    for (auto& e : scene->entities)
        if (e.isTerrain) e.terrainDirty = true;
    hasUnsavedChanges = true;
}

void Editor::redo() {
    if (redoStack.empty()) return;
    undoStack.push_back(*scene);
    *scene = redoStack.back();
    redoStack.pop_back();
    for (auto& e : scene->entities)
        if (e.isTerrain) e.terrainDirty = true;
    hasUnsavedChanges = true;
}
