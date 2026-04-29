#pragma once
#include <string>
#include <cstdint>
#include <glm/glm.hpp>
#include "Scene.hpp"

// Thin wrapper over miniaudio (PIMPL — miniaudio.h is not exposed in this header).
// One AudioEngine instance lives in VulkanEngine.
// Call init() once at startup, cleanup() at shutdown.
// In play mode: onPlayStart() spawns sounds; onPlayStop() stops all.
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    bool init();
    void cleanup();

    // Start sounds for all entities that have hasAudio=true.
    // audioPath: full path to the project's assets/audio/ folder.
    void onPlayStart(const Scene& scene, const std::string& audioPath);

    // Stop and release all playing sounds.
    void onPlayStop();

    // Set the 3D listener (call each frame in play mode).
    void updateListener(glm::vec3 pos, glm::vec3 forward, glm::vec3 up);

    // Update 3D position of a sound for a moving entity.
    void updateEntityPosition(uint32_t entityId, glm::vec3 worldPos);

    // Fire-and-forget one-shot sound (non-spatial).
    void playOneShot(const std::string& filePath);

private:
    struct Impl;
    Impl* pImpl = nullptr;
};
