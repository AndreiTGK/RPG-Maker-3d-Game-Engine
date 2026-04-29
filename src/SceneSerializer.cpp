#include "SceneSerializer.hpp"
#include "json.hpp"
#include <fstream>
#include <iomanip>

using json = nlohmann::json;

void SceneSerializer::save(const Scene& scene, const std::string& fullPath) {
    json j;
    j["sceneName"]      = scene.name;
    j["ambientLight"]   = scene.ambientLight;
    j["sunDirection"]   = scene.sunDirection;
    j["skyColor"]       = scene.skyColor;
    j["shadowsEnabled"] = scene.shadowsEnabled;
    j["nextEntityId"]   = scene.nextEntityId;

    j["entities"] = json::array();
    for (const auto& entity : scene.entities) {
        json e;
        e["id"]       = entity.id;
        e["parentId"] = entity.parentId;
        e["name"]     = entity.name;
        e["visible"]  = entity.visible;
        e["pos"]       = entity.transform.translation;
        e["rot"]       = entity.transform.rotation;
        e["scale"]     = entity.transform.scale;
        e["modelIdx"]  = entity.modelIndex;
        e["texIdx"]    = entity.textureIndex;
        e["metallic"]  = entity.metallic;
        e["roughness"] = entity.roughness;

        // Terrain
        e["isTerrain"]        = entity.isTerrain;
        if (entity.isTerrain) {
            e["terrainWidth"]     = entity.terrainWidth;
            e["terrainDepth"]     = entity.terrainDepth;
            e["terrainAmplitude"] = entity.terrainAmplitude;
            e["terrainFrequency"] = entity.terrainFrequency;
            e["terrainSeed"]      = entity.terrainSeed;
        }

        // Physics
        e["hasCollision"]   = entity.hasCollision;
        e["isStatic"]       = entity.isStatic;
        e["gravityEnabled"] = entity.gravityEnabled;

        // Audio
        e["hasAudio"]     = entity.hasAudio;
        e["audioFile"]    = entity.audioFile;
        e["audioLoop"]    = entity.audioLoop;
        e["audioVolume"]  = entity.audioVolume;
        e["audioMinDist"] = entity.audioMinDist;
        e["audioMaxDist"] = entity.audioMaxDist;

        // Particle emitter
        e["hasEmitter"]           = entity.hasEmitter;
        if (entity.hasEmitter) {
            e["emitterMaxParticles"] = entity.emitterMaxParticles;
            e["emitterRate"]         = entity.emitterRate;
            e["emitterLifetime"]     = entity.emitterLifetime;
            e["emitterSpeed"]        = entity.emitterSpeed;
            e["emitterSpread"]       = entity.emitterSpread;
            e["emitterStartSize"]    = entity.emitterStartSize;
            e["emitterEndSize"]      = entity.emitterEndSize;
            e["emitterStartColor"]   = entity.emitterStartColor;
            e["emitterEndColor"]     = entity.emitterEndColor;
        }

        // AI Agent
        e["hasAIAgent"]       = entity.hasAIAgent;
        if (entity.hasAIAgent) {
            e["aiTargetEntityId"] = entity.aiTargetEntityId;
            e["aiMoveSpeed"]      = entity.aiMoveSpeed;
            e["aiStoppingDist"]   = entity.aiStoppingDist;
        }

        // Skeletal animation
        e["hasSkin"]          = entity.hasSkin;
        if (entity.hasSkin) {
            e["skinModelName"]    = entity.skinModelName;
            e["currentAnimation"] = entity.currentAnimation;
            e["animationLoop"]    = entity.animationLoop;
            e["animationPlaying"] = entity.animationPlaying;
            e["animationSpeed"]   = entity.animationSpeed;
        }

        // Light
        e["isLightSource"]  = entity.isLightSource;
        e["lightIntensity"] = entity.lightIntensity;
        e["lightColor"]     = entity.lightColor;
        e["lightRadius"]    = entity.lightRadius;
        e["lightType"]      = static_cast<int>(entity.lightType);
        e["spotAngle"]      = entity.spotAngle;

        // Lua scripting / trigger / player
        e["luaScriptFile"]    = entity.luaScriptFile;
        e["isTrigger"]        = entity.isTrigger;
        e["isPlayer"]         = entity.isPlayer;
        e["playerCameraMode"] = entity.playerCameraMode;
        e["playerMoveSpeed"]  = entity.playerMoveSpeed;

        // Visual scripting
        e["vsEnabled"] = entity.visualScript.enabled;
        if (entity.visualScript.enabled && !entity.visualScript.blocks.empty()) {
            json blocks = json::array();
            for (const auto& blk : entity.visualScript.blocks) {
                json b;
                b["event"] = (int)blk.event;
                json acts = json::array();
                for (const auto& act : blk.actions) {
                    json a;
                    a["type"] = (int)act.type;
                    json pp = json::array();
                    for (int pi = 0; pi < 5; pi++) pp.push_back(act.p[pi]);
                    a["p"] = pp;
                    acts.push_back(a);
                }
                b["actions"] = acts;
                blocks.push_back(b);
            }
            e["vsBlocks"] = blocks;
        }

        // UI Canvas
        e["hasUICanvas"] = entity.hasUICanvas;
        if (entity.hasUICanvas) {
            json canvas;
            canvas["visible"] = entity.uiCanvas.visible;
            canvas["elements"] = json::array();
            for (const auto& elem : entity.uiCanvas.elements) {
                json el;
                el["type"]       = static_cast<int>(elem.type);
                el["anchor"]     = static_cast<int>(elem.anchor);
                el["offset"]     = elem.offset;
                el["size"]       = elem.size;
                el["text"]       = elem.text;
                el["color"]      = elem.color;
                el["bgColor"]    = elem.bgColor;
                el["value"]      = elem.value;
                el["valueColor"] = elem.valueColor;
                el["texIdx"]     = elem.texIdx;
                canvas["elements"].push_back(el);
            }
            e["uiCanvas"] = canvas;
        }

        j["entities"].push_back(e);
    }

    std::ofstream file(fullPath);
    if (file.is_open())
        file << std::setw(4) << j << '\n';
}

Scene SceneSerializer::load(const std::string& fullPath) {
    Scene scene;
    std::ifstream file(fullPath);
    if (!file.is_open()) return scene;

    json j;
    try { file >> j; } catch (...) { return scene; }

    scene.name           = j.value("sceneName", "Scena Incarcata");
    scene.ambientLight   = j.contains("ambientLight") ? j["ambientLight"].get<glm::vec3>() : glm::vec3(0.2f);
    scene.sunDirection   = j.contains("sunDirection") ? j["sunDirection"].get<glm::vec3>() : glm::vec3(0.5f, 1.0f, 0.3f);
    scene.skyColor       = j.contains("skyColor")     ? j["skyColor"].get<glm::vec3>()     : glm::vec3(0.1f);
    scene.shadowsEnabled = j.value("shadowsEnabled", true);
    scene.nextEntityId   = j.value("nextEntityId", (uint32_t)1);

    if (j.contains("entities")) {
        for (const auto& e : j["entities"]) {
            GameObject obj;
            obj.id                    = e.value("id",       (uint32_t)0);
            obj.parentId              = e.value("parentId", (uint32_t)0);
            obj.name                  = e.value("name", "Entitate");
            obj.visible               = e.value("visible", true);
            obj.transform.translation = e.contains("pos")   ? e["pos"].get<glm::vec3>()   : glm::vec3(0.0f);
            obj.transform.rotation    = e.contains("rot")   ? e["rot"].get<glm::vec3>()   : glm::vec3(0.0f);
            obj.transform.scale       = e.contains("scale") ? e["scale"].get<glm::vec3>() : glm::vec3(1.0f);
            obj.modelIndex            = e.value("modelIdx",  0);
            obj.textureIndex          = e.value("texIdx",    0);
            obj.metallic              = e.value("metallic",  0.0f);
            obj.roughness             = e.value("roughness", 0.5f);

            obj.isTerrain        = e.value("isTerrain",        false);
            obj.terrainWidth     = e.value("terrainWidth",     64);
            obj.terrainDepth     = e.value("terrainDepth",     64);
            obj.terrainAmplitude = e.value("terrainAmplitude", 5.0f);
            obj.terrainFrequency = e.value("terrainFrequency", 0.05f);
            obj.terrainSeed      = e.value("terrainSeed",      42);
            obj.terrainDirty     = true; // always regenerate on load

            // Physics (defaulted for backward compat with old files)
            obj.hasCollision   = e.value("hasCollision",   true);
            obj.isStatic       = e.value("isStatic",       true);
            obj.gravityEnabled = e.value("gravityEnabled", true);

            // Audio
            obj.hasAudio     = e.value("hasAudio",     false);
            obj.audioFile    = e.value("audioFile",    std::string(""));
            obj.audioLoop    = e.value("audioLoop",    false);
            obj.audioVolume  = e.value("audioVolume",  1.0f);
            obj.audioMinDist = e.value("audioMinDist", 1.0f);
            obj.audioMaxDist = e.value("audioMaxDist", 20.0f);

            // Particle emitter
            obj.hasEmitter          = e.value("hasEmitter", false);
            obj.emitterMaxParticles = e.value("emitterMaxParticles", 100);
            obj.emitterRate         = e.value("emitterRate",         20.0f);
            obj.emitterLifetime     = e.value("emitterLifetime",     2.0f);
            obj.emitterSpeed        = e.value("emitterSpeed",        3.0f);
            obj.emitterSpread       = e.value("emitterSpread",       0.5f);
            obj.emitterStartSize    = e.value("emitterStartSize",    0.3f);
            obj.emitterEndSize      = e.value("emitterEndSize",      0.0f);
            obj.emitterStartColor   = e.contains("emitterStartColor") ? e["emitterStartColor"].get<glm::vec4>() : glm::vec4(1.0f, 0.8f, 0.3f, 1.0f);
            obj.emitterEndColor     = e.contains("emitterEndColor")   ? e["emitterEndColor"].get<glm::vec4>()   : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

            // AI Agent
            obj.hasAIAgent       = e.value("hasAIAgent",       false);
            obj.aiTargetEntityId = e.value("aiTargetEntityId", (uint32_t)0);
            obj.aiMoveSpeed      = e.value("aiMoveSpeed",      3.0f);
            obj.aiStoppingDist   = e.value("aiStoppingDist",   0.5f);

            // Skeletal animation
            obj.hasSkin          = e.value("hasSkin",          false);
            obj.skinModelName    = e.value("skinModelName",    std::string(""));
            obj.currentAnimation = e.value("currentAnimation", 0);
            obj.animationLoop    = e.value("animationLoop",    true);
            obj.animationPlaying = e.value("animationPlaying", false);
            obj.animationSpeed   = e.value("animationSpeed",   1.0f);

            // Light
            obj.isLightSource  = e.value("isLightSource",  false);
            obj.lightIntensity = e.value("lightIntensity", 5.0f);
            obj.lightColor     = e.contains("lightColor") ? e["lightColor"].get<glm::vec3>() : glm::vec3(1.0f);
            obj.lightRadius    = e.value("lightRadius",    10.0f);
            obj.lightType      = static_cast<LightType>(e.value("lightType", 0));
            obj.spotAngle      = e.value("spotAngle",      30.0f);

            // Lua scripting / trigger / player
            obj.luaScriptFile    = e.value("luaScriptFile",    std::string(""));
            obj.isTrigger        = e.value("isTrigger",        false);
            obj.isPlayer         = e.value("isPlayer",         false);
            obj.playerCameraMode = e.value("playerCameraMode", 0);
            obj.playerMoveSpeed  = e.value("playerMoveSpeed",  5.0f);

            // Visual scripting
            obj.visualScript.enabled = e.value("vsEnabled", false);
            if (e.contains("vsBlocks")) {
                for (const auto& b : e["vsBlocks"]) {
                    VSBlock blk;
                    blk.event = static_cast<VSEventType>(b.value("event", 0));
                    if (b.contains("actions")) {
                        for (const auto& a : b["actions"]) {
                            VSAction act;
                            act.type = static_cast<VSActionType>(a.value("type", 0));
                            if (a.contains("p")) {
                                const auto& pp = a["p"];
                                for (int pi = 0; pi < 5 && pi < (int)pp.size(); pi++)
                                    act.p[pi] = pp[pi].get<std::string>();
                            }
                            blk.actions.push_back(act);
                        }
                    }
                    obj.visualScript.blocks.push_back(blk);
                }
            }

            // UI Canvas
            obj.hasUICanvas = e.value("hasUICanvas", false);
            if (obj.hasUICanvas && e.contains("uiCanvas")) {
                const auto& c = e["uiCanvas"];
                obj.uiCanvas.visible = c.value("visible", true);
                if (c.contains("elements")) {
                    for (const auto& el : c["elements"]) {
                        UIElement elem;
                        elem.type       = static_cast<UIElementType>(el.value("type", 0));
                        elem.anchor     = static_cast<UIAnchor>(el.value("anchor", 0));
                        elem.offset     = el.contains("offset")     ? el["offset"].get<glm::vec2>()     : glm::vec2(10.0f, 10.0f);
                        elem.size       = el.contains("size")       ? el["size"].get<glm::vec2>()       : glm::vec2(200.0f, 30.0f);
                        elem.text       = el.value("text", std::string("Label"));
                        elem.color      = el.contains("color")      ? el["color"].get<glm::vec4>()      : glm::vec4(1.0f);
                        elem.bgColor    = el.contains("bgColor")    ? el["bgColor"].get<glm::vec4>()    : glm::vec4(0.0f, 0.0f, 0.0f, 0.6f);
                        elem.value      = el.value("value", 1.0f);
                        elem.valueColor = el.contains("valueColor") ? el["valueColor"].get<glm::vec4>() : glm::vec4(0.2f, 0.8f, 0.2f, 1.0f);
                        elem.texIdx     = el.value("texIdx", -1);
                        obj.uiCanvas.elements.push_back(elem);
                    }
                }
            }

            // Backward compat: assign stable IDs to entities that were saved before hierarchy was added
            if (obj.id == 0) {
                obj.id = scene.nextEntityId++;
            }

            scene.entities.push_back(obj);
        }
    }

    // Ensure nextEntityId is beyond any loaded IDs (handles files with gaps)
    for (const auto& obj : scene.entities) {
        if (obj.id >= scene.nextEntityId)
            scene.nextEntityId = obj.id + 1;
    }

    return scene;
}
