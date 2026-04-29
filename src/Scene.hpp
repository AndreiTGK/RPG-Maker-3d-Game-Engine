#pragma once
#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include <unordered_set>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "json.hpp" // NOU

// ---------------------------------------------------------------------------
// In-game UI data structures (play-mode overlay, not ImGui)
// ---------------------------------------------------------------------------
enum class UIElementType { Label = 0, Button = 1, Image = 2, Healthbar = 3 };
enum class UIAnchor      { TopLeft = 0, TopCenter = 1, TopRight = 2,
                           MiddleLeft = 3, Center = 4, MiddleRight = 5,
                           BottomLeft = 6, BottomCenter = 7, BottomRight = 8 };

struct UIElement {
    UIElementType type      = UIElementType::Label;
    UIAnchor      anchor    = UIAnchor::TopLeft;
    glm::vec2     offset    = {10.0f, 10.0f};   // pixels from anchor
    glm::vec2     size      = {200.0f, 30.0f};  // pixels (w, h)
    std::string   text      = "Label";
    glm::vec4     color     = {1.0f, 1.0f, 1.0f, 1.0f}; // text / image tint
    glm::vec4     bgColor   = {0.0f, 0.0f, 0.0f, 0.6f}; // background rect
    float         value     = 1.0f;              // healthbar fill fraction [0,1]
    glm::vec4     valueColor= {0.2f, 0.8f, 0.2f, 1.0f}; // healthbar fill color
    int           texIdx    = -1;                // for Image: scene texture index
    // Runtime-only — NOT serialized
    std::function<void()> onClick;
};

struct UICanvas {
    bool                    visible  = true;
    std::vector<UIElement>  elements;
};

using json = nlohmann::json;
namespace glm {
    inline void to_json(nlohmann::json& j, const vec2& v) {
        j = nlohmann::json{{"x", v.x}, {"y", v.y}};
    }
    inline void from_json(const nlohmann::json& j, vec2& v) {
        v.x = j.value("x", 0.0f); v.y = j.value("y", 0.0f);
    }
    inline void to_json(nlohmann::json& j, const vec3& v) {
        j = nlohmann::json{{"x", v.x}, {"y", v.y}, {"z", v.z}};
    }
    inline void from_json(const nlohmann::json& j, vec3& v) {
        v.x = j.at("x").get<float>();
        v.y = j.at("y").get<float>();
        v.z = j.at("z").get<float>();
    }
    inline void to_json(nlohmann::json& j, const vec4& v) {
        j = nlohmann::json{{"r", v.r}, {"g", v.g}, {"b", v.b}, {"a", v.a}};
    }
    inline void from_json(const nlohmann::json& j, vec4& v) {
        v.r = j.value("r", 1.0f); v.g = j.value("g", 1.0f);
        v.b = j.value("b", 1.0f); v.a = j.value("a", 1.0f);
    }
}

struct TransformComponent {
    glm::vec3 translation{0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    glm::vec3 rotation{0.0f, 0.0f, 0.0f};

    glm::mat4 mat4() const {
        glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation);
        transform = glm::rotate(transform, rotation.z, {0.0f, 0.0f, 1.0f});
        transform = glm::rotate(transform, rotation.y, {0.0f, 1.0f, 0.0f});
        transform = glm::rotate(transform, rotation.x, {1.0f, 0.0f, 0.0f});
        transform = glm::scale(transform, scale);
        return transform;
    }
};

enum class LightType : int { Point = 0, Spot = 1 };

// ─── Visual Scripting ────────────────────────────────────────────────────────
enum class VSEventType  { OnStart=0, OnUpdate=1, OnInteract=2,
                          OnTriggerEnter=3, OnTriggerExit=4 };
enum class VSActionType { Log=0, PlaySound=1, SetPosition=2, MoveToward=3,
                          SetVisible=4, SetRotation=5, LoadScene=6,
                          ShowDialogue=7, Wait=8, SetUIText=9, SetUIValue=10 };

struct VSAction {
    VSActionType              type = VSActionType::Log;
    std::array<std::string,5> p    = {};  // param slots (varies by type)
};

struct VSBlock {
    VSEventType           event = VSEventType::OnStart;
    std::vector<VSAction> actions;
};

struct VisualScript {
    bool                  enabled = false;
    std::vector<VSBlock>  blocks;
};

struct GameObject {
    uint32_t  id       = 0;         // unique stable ID (0 = uninitialized)
    uint32_t  parentId = 0;         // 0 = no parent (root)

    std::string name = "Obiect_Nou";
    TransformComponent transform;   // local-space transform
    bool visible = true;            // false = skipped by renderScene and RT

    int modelIndex = 0;
    int textureIndex = 0;

    // PBR material properties
    float metallic  = 0.0f;   // 0 = dielectric, 1 = fully metallic
    float roughness = 0.5f;   // 0 = mirror, 1 = fully diffuse

    // Terrain
    bool  isTerrain          = false;
    int   terrainWidth       = 64;    // quads in X
    int   terrainDepth       = 64;    // quads in Y
    float terrainAmplitude   = 5.0f;
    float terrainFrequency   = 0.05f;
    int   terrainSeed        = 42;
    // Runtime-only: set by engine when params change; NOT serialized
    bool  terrainDirty       = true;

    bool hasCollision    = true;
    bool isStatic        = true;
    bool gravityEnabled  = true;   // affected by gravity when not static

    // Runtime physics state — NOT serialized, reset on play start
    glm::vec3 velocity   = {0.0f, 0.0f, 0.0f};

    // Event callbacks — set by scripting or play-mode systems, NOT serialized
    std::function<void()>                        onStart;
    std::function<void(float dt)>                onUpdate;
    std::function<void(GameObject* other)>       onCollision;

    // --- Audio source ---
    bool        hasAudio     = false;
    std::string audioFile    = "";      // filename relative to assets/audio/
    bool        audioLoop    = false;
    float       audioVolume  = 1.0f;
    float       audioMinDist = 1.0f;   // full volume within this distance
    float       audioMaxDist = 20.0f;  // silence beyond this distance

    // --- Particle emitter ---
    bool        hasEmitter          = false;
    int         emitterMaxParticles = 100;
    float       emitterRate         = 20.0f;      // particles/sec
    float       emitterLifetime     = 2.0f;       // seconds per particle
    float       emitterSpeed        = 3.0f;       // initial speed (m/s)
    float       emitterSpread       = 0.5f;       // cone half-angle (radians)
    float       emitterStartSize    = 0.3f;
    float       emitterEndSize      = 0.0f;
    glm::vec4   emitterStartColor   = {1.0f, 0.8f, 0.3f, 1.0f};
    glm::vec4   emitterEndColor     = {1.0f, 0.0f, 0.0f, 0.0f};

    // --- AI Agent ---
    bool     hasAIAgent       = false;
    uint32_t aiTargetEntityId = 0;       // 0 = no target
    float    aiMoveSpeed      = 3.0f;    // world units/sec
    float    aiStoppingDist   = 0.5f;    // stop within this distance of target

    // --- Skeletal animation ---
    bool        hasSkin          = false;
    std::string skinModelName    = "";    // GLTF/GLB file name in assets/models/
    int         currentAnimation = 0;
    bool        animationLoop    = true;
    bool        animationPlaying = false;
    float       animationSpeed   = 1.0f;

    // --- Light state ---
    bool      isLightSource  = false;
    float     lightIntensity = 5.0f;
    glm::vec3 lightColor     = {1.0f, 1.0f, 1.0f};
    float     lightRadius    = 10.0f;
    LightType lightType      = LightType::Point;
    float     spotAngle      = 30.0f;   // degrees, half-angle (Spot only)

    // --- In-game UI canvas ---
    bool     hasUICanvas = false;
    UICanvas uiCanvas;

    // --- Lua scripting ---
    std::string luaScriptFile = "";   // relative to project scripts/ dir ("" = none)

    // --- Trigger zone (isTrigger: overlap events, no physics response) ---
    bool isTrigger = false;

    // --- Player controller ---
    bool  isPlayer         = false;
    int   playerCameraMode = 0;       // 0=FreeFly, 1=ThirdPerson, 2=TopDown
    float playerMoveSpeed  = 5.0f;

    // --- Visual scripting ---
    VisualScript visualScript;
};

struct Scene {
    std::string name = "Noua_Scena";
    std::vector<GameObject> entities;
    glm::vec3 ambientLight  = {0.2f, 0.2f, 0.2f};
    glm::vec3 sunDirection  = {0.5f, 1.0f, 0.3f};
    glm::vec3 skyColor      = {0.1f, 0.1f, 0.1f};
    bool shadowsEnabled     = true;
    uint32_t  nextEntityId  = 1;    // monotonic ID counter
};

// Returns the accumulated world-space transform for the entity at entities[index].
// Iterative parent-chain walk; breaks on cycle or missing parent.
inline glm::mat4 getWorldTransform(const Scene& scene, int index) {
    if (index < 0 || index >= (int)scene.entities.size())
        return glm::mat4(1.0f);

    // Collect ancestor chain bottom-up, stopping at root or cycle
    std::vector<int> chain;
    std::unordered_set<uint32_t> visited;
    int cur = index;
    while (cur >= 0 && cur < (int)scene.entities.size()) {
        uint32_t id = scene.entities[cur].id;
        if (visited.count(id)) break;
        visited.insert(id);
        chain.push_back(cur);
        uint32_t parentId = scene.entities[cur].parentId;
        if (parentId == 0) break;
        cur = -1;
        for (int i = 0; i < (int)scene.entities.size(); i++) {
            if (scene.entities[i].id == parentId) { cur = i; break; }
        }
    }

    // Multiply root→entity
    glm::mat4 result(1.0f);
    for (int i = (int)chain.size() - 1; i >= 0; i--)
        result = result * scene.entities[chain[i]].transform.mat4();
    return result;
}
