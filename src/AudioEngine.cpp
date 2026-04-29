#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "AudioEngine.hpp"
#include "EngineLog.hpp"
#include <unordered_map>

struct AudioEngine::Impl {
    ma_engine engine{};
    bool      initialized = false;
    std::unordered_map<uint32_t, ma_sound*> sounds;
};

AudioEngine::AudioEngine() : pImpl(new Impl()) {}

AudioEngine::~AudioEngine() {
    cleanup();
    delete pImpl;
}

bool AudioEngine::init() {
    if (pImpl->initialized) return true;
    ma_engine_config cfg = ma_engine_config_init();
    cfg.listenerCount = 1;
    if (ma_engine_init(&cfg, &pImpl->engine) != MA_SUCCESS) {
        LOG_ERROR("AudioEngine: failed to initialise miniaudio engine");
        return false;
    }
    pImpl->initialized = true;
    LOG_INFO("AudioEngine: initialised");
    return true;
}

void AudioEngine::cleanup() {
    if (!pImpl->initialized) return;
    onPlayStop();
    ma_engine_uninit(&pImpl->engine);
    pImpl->initialized = false;
}

void AudioEngine::onPlayStart(const Scene& scene, const std::string& audioPath) {
    if (!pImpl->initialized) return;
    onPlayStop(); // clear any leftover sounds first

    for (int i = 0; i < (int)scene.entities.size(); i++) {
        const auto& entity = scene.entities[i];
        if (!entity.hasAudio || entity.audioFile.empty()) continue;

        std::string filePath = audioPath + "/" + entity.audioFile;
        glm::vec3   worldPos = glm::vec3(getWorldTransform(scene, i)[3]);

        ma_sound* sound = new ma_sound;
        ma_uint32 flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_ASYNC;
        if (ma_sound_init_from_file(&pImpl->engine, filePath.c_str(), flags,
                                     nullptr, nullptr, sound) != MA_SUCCESS) {
            LOG_WARNING("AudioEngine: cannot load '%s'", filePath.c_str());
            delete sound;
            continue;
        }

        ma_sound_set_looping(sound, entity.audioLoop ? MA_TRUE : MA_FALSE);
        ma_sound_set_volume(sound, entity.audioVolume);
        ma_sound_set_spatialization_enabled(sound, MA_TRUE);
        ma_sound_set_position(sound, worldPos.x, worldPos.y, worldPos.z);
        ma_sound_set_min_distance(sound, entity.audioMinDist);
        ma_sound_set_max_distance(sound, entity.audioMaxDist);
        ma_sound_start(sound);

        pImpl->sounds[entity.id] = sound;
        LOG_INFO("AudioEngine: playing '%s' (entity %u)", entity.audioFile.c_str(), entity.id);
    }
}

void AudioEngine::onPlayStop() {
    for (auto& [id, sound] : pImpl->sounds) {
        ma_sound_uninit(sound);
        delete sound;
    }
    pImpl->sounds.clear();
}

void AudioEngine::updateListener(glm::vec3 pos, glm::vec3 forward, glm::vec3 up) {
    if (!pImpl->initialized) return;
    ma_engine_listener_set_position (&pImpl->engine, 0, pos.x,     pos.y,     pos.z);
    ma_engine_listener_set_direction(&pImpl->engine, 0, forward.x, forward.y, forward.z);
    ma_engine_listener_set_world_up (&pImpl->engine, 0, up.x,      up.y,      up.z);
}

void AudioEngine::updateEntityPosition(uint32_t entityId, glm::vec3 worldPos) {
    if (!pImpl->initialized) return;
    auto it = pImpl->sounds.find(entityId);
    if (it != pImpl->sounds.end())
        ma_sound_set_position(it->second, worldPos.x, worldPos.y, worldPos.z);
}

void AudioEngine::playOneShot(const std::string& filePath) {
    if (!pImpl->initialized) return;
    ma_engine_play_sound(&pImpl->engine, filePath.c_str(), nullptr);
}
