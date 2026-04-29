#pragma once
#include "Scene.hpp"
#include "Camera.hpp"
#include "AudioEngine.hpp"
#include "LuaEngine.hpp"
#include "NavMesh.hpp"
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct AIAgentState {
    std::vector<glm::vec2> path;
    glm::vec3 lastTargetPos{0};
    float     repathTimer  = 0.0f;
};

// ---------------------------------------------------------------------------
// Runtime — owns all gameplay systems and the play/stop lifecycle.
// Operates on a Scene* provided by VulkanEngine (activeScene stays there
// so rendering code needs no changes).
// ---------------------------------------------------------------------------

class Runtime {
public:
    // Call once at init; scene must outlive Runtime.
    void init(Scene* scene);

    // Enter play mode: saves editor scene/camera, resets physics, starts
    // audio + Lua. nm is a pre-built navmesh from VulkanEngine.
    void start(Camera& cam,
               const std::string& scriptsPath,
               const std::string& audioPath,
               Navmesh nm);

    // Exit play mode: stops Lua/audio, restores scene and camera.
    void stop(Camera& cam);

    // Tick all gameplay systems for one frame.
    void tick(float dt, Camera& cam);

    bool isPlaying() const { return playing; }

    // Call from VulkanEngine when a scene is loaded/switched outside of play mode
    // to discard stale per-entity runtime state.
    void clearSceneState() { aiAgentStates.clear(); activeTriggerOverlaps.clear(); }

    // Input callbacks — set by VulkanEngine before start().
    std::function<bool(int glfwKey)>   cbIsKeyDown;

    // Subsystems — public so VulkanEngine can wire Lua callbacks and
    // forward dialogue state to renderDialogue().
    LuaEngine   luaEngine;
    AudioEngine audioEngine;

private:
    Scene* scene    = nullptr;
    Scene  savedScene;
    Camera savedCamera;
    bool   playing  = false;

    Navmesh                          navmesh;
    std::map<uint32_t, AIAgentState> aiAgentStates;
    std::set<uint64_t>               activeTriggerOverlaps;
    bool                             wasInteractPressed = false;

    void tickPhysics(float dt);
    void tickPlayerController(float dt, Camera& cam);
    void tickTriggers();
    void tickInteract();
    void tickAI(float dt);
};
