#pragma once
#include "Scene.hpp"
#include "json.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <iomanip>
#include <filesystem>

using json = nlohmann::json;

namespace PrefabSerializer {

// ---------------------------------------------------------------------------
// Internal helpers — entity ↔ JSON (same field layout as SceneSerializer)
// ---------------------------------------------------------------------------

static json entityToJson(const GameObject& entity) {
    json e;
    e["id"]       = entity.id;
    e["parentId"] = entity.parentId;
    e["name"]     = entity.name;
    e["visible"]  = entity.visible;
    e["pos"]      = entity.transform.translation;
    e["rot"]      = entity.transform.rotation;
    e["scale"]    = entity.transform.scale;
    e["modelIdx"] = entity.modelIndex;
    e["texIdx"]   = entity.textureIndex;
    e["metallic"] = entity.metallic;
    e["roughness"]= entity.roughness;

    e["isTerrain"] = entity.isTerrain;
    if (entity.isTerrain) {
        e["terrainWidth"]     = entity.terrainWidth;
        e["terrainDepth"]     = entity.terrainDepth;
        e["terrainAmplitude"] = entity.terrainAmplitude;
        e["terrainFrequency"] = entity.terrainFrequency;
        e["terrainSeed"]      = entity.terrainSeed;
    }

    e["hasCollision"]   = entity.hasCollision;
    e["isStatic"]       = entity.isStatic;
    e["gravityEnabled"] = entity.gravityEnabled;

    e["hasAudio"]    = entity.hasAudio;
    e["audioFile"]   = entity.audioFile;
    e["audioLoop"]   = entity.audioLoop;
    e["audioVolume"] = entity.audioVolume;
    e["audioMinDist"]= entity.audioMinDist;
    e["audioMaxDist"]= entity.audioMaxDist;

    e["hasEmitter"] = entity.hasEmitter;
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

    e["hasAIAgent"] = entity.hasAIAgent;
    if (entity.hasAIAgent) {
        e["aiTargetEntityId"] = entity.aiTargetEntityId;
        e["aiMoveSpeed"]      = entity.aiMoveSpeed;
        e["aiStoppingDist"]   = entity.aiStoppingDist;
    }

    e["hasSkin"] = entity.hasSkin;
    if (entity.hasSkin) {
        e["skinModelName"]    = entity.skinModelName;
        e["currentAnimation"] = entity.currentAnimation;
        e["animationLoop"]    = entity.animationLoop;
        e["animationPlaying"] = entity.animationPlaying;
        e["animationSpeed"]   = entity.animationSpeed;
    }

    e["isLightSource"]  = entity.isLightSource;
    e["lightIntensity"] = entity.lightIntensity;
    e["lightColor"]     = entity.lightColor;
    e["lightRadius"]    = entity.lightRadius;
    e["lightType"]      = static_cast<int>(entity.lightType);
    e["spotAngle"]      = entity.spotAngle;

    e["luaScriptFile"]    = entity.luaScriptFile;
    e["isTrigger"]        = entity.isTrigger;
    e["isPlayer"]         = entity.isPlayer;
    e["playerCameraMode"] = entity.playerCameraMode;
    e["playerMoveSpeed"]  = entity.playerMoveSpeed;

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

    e["hasUICanvas"] = entity.hasUICanvas;
    if (entity.hasUICanvas) {
        json canvas;
        canvas["visible"] = entity.uiCanvas.visible;
        canvas["elements"] = json::array();
        for (const auto& elem : entity.uiCanvas.elements) {
            json el;
            el["type"]      = static_cast<int>(elem.type);
            el["anchor"]    = static_cast<int>(elem.anchor);
            el["offset"]    = elem.offset;
            el["size"]      = elem.size;
            el["text"]      = elem.text;
            el["color"]     = elem.color;
            el["bgColor"]   = elem.bgColor;
            el["value"]     = elem.value;
            el["valueColor"]= elem.valueColor;
            el["texIdx"]    = elem.texIdx;
            canvas["elements"].push_back(el);
        }
        e["uiCanvas"] = canvas;
    }

    return e;
}

static GameObject entityFromJson(const json& e) {
    GameObject obj;
    obj.id                    = e.value("id",       (uint32_t)0);
    obj.parentId              = e.value("parentId", (uint32_t)0);
    obj.name                  = e.value("name", "Prefab");
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
    obj.terrainDirty     = obj.isTerrain;

    obj.hasCollision   = e.value("hasCollision",   true);
    obj.isStatic       = e.value("isStatic",       true);
    obj.gravityEnabled = e.value("gravityEnabled", true);

    obj.hasAudio     = e.value("hasAudio",     false);
    obj.audioFile    = e.value("audioFile",    std::string(""));
    obj.audioLoop    = e.value("audioLoop",    false);
    obj.audioVolume  = e.value("audioVolume",  1.0f);
    obj.audioMinDist = e.value("audioMinDist", 1.0f);
    obj.audioMaxDist = e.value("audioMaxDist", 20.0f);

    obj.hasEmitter          = e.value("hasEmitter", false);
    obj.emitterMaxParticles = e.value("emitterMaxParticles", 100);
    obj.emitterRate         = e.value("emitterRate",         20.0f);
    obj.emitterLifetime     = e.value("emitterLifetime",     2.0f);
    obj.emitterSpeed        = e.value("emitterSpeed",        3.0f);
    obj.emitterSpread       = e.value("emitterSpread",       0.5f);
    obj.emitterStartSize    = e.value("emitterStartSize",    0.3f);
    obj.emitterEndSize      = e.value("emitterEndSize",      0.0f);
    obj.emitterStartColor   = e.contains("emitterStartColor") ? e["emitterStartColor"].get<glm::vec4>() : glm::vec4(1.0f,0.8f,0.3f,1.0f);
    obj.emitterEndColor     = e.contains("emitterEndColor")   ? e["emitterEndColor"].get<glm::vec4>()   : glm::vec4(1.0f,0.0f,0.0f,0.0f);

    obj.hasAIAgent       = e.value("hasAIAgent",       false);
    obj.aiTargetEntityId = e.value("aiTargetEntityId", (uint32_t)0);
    obj.aiMoveSpeed      = e.value("aiMoveSpeed",      3.0f);
    obj.aiStoppingDist   = e.value("aiStoppingDist",   0.5f);

    obj.hasSkin          = e.value("hasSkin",          false);
    obj.skinModelName    = e.value("skinModelName",    std::string(""));
    obj.currentAnimation = e.value("currentAnimation", 0);
    obj.animationLoop    = e.value("animationLoop",    true);
    obj.animationPlaying = e.value("animationPlaying", false);
    obj.animationSpeed   = e.value("animationSpeed",   1.0f);

    obj.isLightSource  = e.value("isLightSource",  false);
    obj.lightIntensity = e.value("lightIntensity", 5.0f);
    obj.lightColor     = e.contains("lightColor") ? e["lightColor"].get<glm::vec3>() : glm::vec3(1.0f);
    obj.lightRadius    = e.value("lightRadius",    10.0f);
    obj.lightType      = static_cast<LightType>(e.value("lightType", 0));
    obj.spotAngle      = e.value("spotAngle",      30.0f);

    obj.luaScriptFile    = e.value("luaScriptFile",    std::string(""));
    obj.isTrigger        = e.value("isTrigger",        false);
    obj.isPlayer         = e.value("isPlayer",         false);
    obj.playerCameraMode = e.value("playerCameraMode", 0);
    obj.playerMoveSpeed  = e.value("playerMoveSpeed",  5.0f);

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
                        int pi = 0;
                        for (const auto& pv : a["p"]) { if (pi < 5) act.p[pi++] = pv.get<std::string>(); }
                    }
                    blk.actions.push_back(act);
                }
            }
            obj.visualScript.blocks.push_back(blk);
        }
    }

    obj.hasUICanvas = e.value("hasUICanvas", false);
    if (obj.hasUICanvas && e.contains("uiCanvas")) {
        const auto& jc = e["uiCanvas"];
        obj.uiCanvas.visible = jc.value("visible", true);
        if (jc.contains("elements")) {
            for (const auto& el : jc["elements"]) {
                UIElement elem;
                elem.type      = static_cast<UIElementType>(el.value("type", 0));
                elem.anchor    = static_cast<UIAnchor>(el.value("anchor", 0));
                elem.offset    = el.contains("offset") ? el["offset"].get<glm::vec2>() : glm::vec2(0.0f);
                elem.size      = el.contains("size")   ? el["size"].get<glm::vec2>()   : glm::vec2(100.0f, 30.0f);
                elem.text      = el.value("text",  std::string(""));
                elem.color     = el.contains("color")      ? el["color"].get<glm::vec4>()      : glm::vec4(1.0f);
                elem.bgColor   = el.contains("bgColor")    ? el["bgColor"].get<glm::vec4>()    : glm::vec4(0,0,0,0.5f);
                elem.value     = el.value("value", 1.0f);
                elem.valueColor= el.contains("valueColor") ? el["valueColor"].get<glm::vec4>() : glm::vec4(0,1,0,1);
                elem.texIdx    = el.value("texIdx", -1);
                obj.uiCanvas.elements.push_back(elem);
            }
        }
    }

    return obj;
}

// ---------------------------------------------------------------------------
// Collect entity at rootIdx and all of its descendants
// ---------------------------------------------------------------------------
static std::vector<int> collectSubtree(const Scene& scene, int rootIdx) {
    std::vector<int> result;
    std::vector<uint32_t> frontier = { scene.entities[rootIdx].id };
    // seed with root
    result.push_back(rootIdx);
    while (!frontier.empty()) {
        std::vector<uint32_t> next;
        for (int i = 0; i < (int)scene.entities.size(); i++) {
            const auto& e = scene.entities[i];
            bool already = false;
            for (int r : result) if (r == i) { already = true; break; }
            if (already) continue;
            for (uint32_t pid : frontier) {
                if (e.parentId == pid) { result.push_back(i); next.push_back(e.id); break; }
            }
        }
        frontier = next;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Save: serialise entity subtree to a .prefab file
// ---------------------------------------------------------------------------
static bool save(const Scene& scene, int rootIdx, const std::string& path) {
    if (rootIdx < 0 || rootIdx >= (int)scene.entities.size()) return false;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::vector<int> indices = collectSubtree(scene, rootIdx);
    json j;
    j["entities"] = json::array();
    for (int idx : indices)
        j["entities"].push_back(entityToJson(scene.entities[idx]));

    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << std::setw(4) << j << '\n';
    return true;
}

// ---------------------------------------------------------------------------
// Instantiate: load prefab, remap IDs, append to scene
// Returns the index of the first (root) entity added, or -1 on failure.
// ---------------------------------------------------------------------------
static int instantiate(Scene& scene, const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return -1;
    json j;
    try { f >> j; } catch (...) { return -1; }
    if (!j.contains("entities") || j["entities"].empty()) return -1;

    std::vector<GameObject> loaded;
    for (const auto& e : j["entities"])
        loaded.push_back(entityFromJson(e));

    // Build old-id → new-id remap
    std::map<uint32_t, uint32_t> idMap;
    for (auto& obj : loaded) {
        uint32_t newId = scene.nextEntityId++;
        idMap[obj.id] = newId;
        obj.id = newId;
    }
    // Remap parentIds; root's old parentId points outside the prefab → set to 0
    uint32_t rootOldParent = loaded[0].parentId; // original parent of root (outside prefab)
    for (auto& obj : loaded) {
        if (obj.parentId == 0 || idMap.find(obj.parentId) == idMap.end())
            obj.parentId = 0; // detach from original scene parent
        else
            obj.parentId = idMap[obj.parentId];
    }
    (void)rootOldParent;

    int firstIdx = (int)scene.entities.size();
    for (auto& obj : loaded)
        scene.entities.push_back(obj);
    return firstIdx;
}

} // namespace PrefabSerializer
