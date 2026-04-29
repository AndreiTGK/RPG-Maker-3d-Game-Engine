#include "Runtime.hpp"
#include "Physics.hpp"
#include "EngineLog.hpp"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <algorithm>
#include <queue>
#include <cmath>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void Runtime::init(Scene* s) {
    scene = s;
    luaEngine.init(s);
    audioEngine.init();
}

void Runtime::start(Camera& cam,
                    const std::string& scriptsPath,
                    const std::string& audioPath,
                    Navmesh nm) {
    savedScene  = *scene;
    savedCamera = cam;
    navmesh     = std::move(nm);
    aiAgentStates.clear();
    activeTriggerOverlaps.clear();
    wasInteractPressed = false;

    for (auto& e : scene->entities)
        e.velocity = glm::vec3(0.0f);

    audioEngine.onPlayStart(*scene, audioPath);
    luaEngine.onPlayStart(scriptsPath);
    playing = true;
    LOG_INFO("Play mode started.");
}

void Runtime::stop(Camera& cam) {
    luaEngine.onPlayStop();
    audioEngine.onPlayStop();
    *scene = savedScene;
    cam    = savedCamera;
    aiAgentStates.clear();
    activeTriggerOverlaps.clear();
    navmesh            = Navmesh{};
    wasInteractPressed = false;
    playing            = false;
    LOG_INFO("Play mode stopped — scene restored.");
}

void Runtime::tick(float dt, Camera& cam) {
    tickPlayerController(dt, cam);
    tickPhysics(dt);
    tickTriggers();
    tickAI(dt);
    luaEngine.tick(dt);
    tickInteract();
    audioEngine.updateListener(cam.pos, cam.front, cam.up);
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

void Runtime::tickPhysics(float dt) {
    static constexpr float kGravity = 9.81f;
    static constexpr float kGroundZ = 0.0f;

    for (int i = 0; i < (int)scene->entities.size(); i++) {
        auto& e = scene->entities[i];
        if (e.onUpdate) e.onUpdate(dt);
        if (e.isStatic || !e.hasCollision) continue;

        if (e.gravityEnabled)
            e.velocity.z -= kGravity * dt;

        e.transform.translation += e.velocity * dt;

        float halfH = e.transform.scale.z * 0.5f;
        if (e.transform.translation.z - halfH < kGroundZ) {
            e.transform.translation.z = kGroundZ + halfH;
            if (e.velocity.z < 0.0f) e.velocity.z = 0.0f;
        }
    }

    for (int i = 0; i < (int)scene->entities.size(); i++) {
        auto& a = scene->entities[i];
        if (a.isStatic || !a.hasCollision) continue;

        for (int j = 0; j < (int)scene->entities.size(); j++) {
            if (i == j) continue;
            auto& b = scene->entities[j];
            if (!b.hasCollision) continue;
            if (!PhysicsEngine::checkCollision(a, b)) continue;

            if (i < j) {
                if (a.onCollision) a.onCollision(&b);
                if (b.onCollision) b.onCollision(&a);
            }

            glm::vec3 aMin = a.transform.translation - a.transform.scale * 0.5f;
            glm::vec3 aMax = a.transform.translation + a.transform.scale * 0.5f;
            glm::vec3 bMin = b.transform.translation - b.transform.scale * 0.5f;
            glm::vec3 bMax = b.transform.translation + b.transform.scale * 0.5f;

            float ox = glm::max(0.0f, glm::min(aMax.x - bMin.x, bMax.x - aMin.x));
            float oy = glm::max(0.0f, glm::min(aMax.y - bMin.y, bMax.y - aMin.y));
            float oz = glm::max(0.0f, glm::min(aMax.z - bMin.z, bMax.z - aMin.z));

            if (ox <= oy && ox <= oz) {
                float sign = (a.transform.translation.x < b.transform.translation.x) ? -1.0f : 1.0f;
                a.transform.translation.x += sign * ox;
                a.velocity.x = 0.0f;
            } else if (oy <= ox && oy <= oz) {
                float sign = (a.transform.translation.y < b.transform.translation.y) ? -1.0f : 1.0f;
                a.transform.translation.y += sign * oy;
                a.velocity.y = 0.0f;
            } else {
                float sign = (a.transform.translation.z < b.transform.translation.z) ? -1.0f : 1.0f;
                a.transform.translation.z += sign * oz;
                a.velocity.z = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Player controller
// ---------------------------------------------------------------------------

void Runtime::tickPlayerController(float dt, Camera& cam) {
    if (!cbIsKeyDown) return;

    int playerIdx = -1;
    for (int i = 0; i < (int)scene->entities.size(); i++) {
        if (scene->entities[i].isPlayer) { playerIdx = i; break; }
    }
    if (playerIdx < 0) return;

    auto& player = scene->entities[playerIdx];
    float speed  = player.playerMoveSpeed * dt;

    glm::vec3 fwd = cam.front;
    fwd.z = 0.0f;
    if (glm::length(fwd) > 0.001f) fwd = glm::normalize(fwd);
    else fwd = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 0.0f, 1.0f)));

    glm::vec3 move(0.0f);
    if (cbIsKeyDown(GLFW_KEY_W)) move += fwd;
    if (cbIsKeyDown(GLFW_KEY_S)) move -= fwd;
    if (cbIsKeyDown(GLFW_KEY_A)) move -= right;
    if (cbIsKeyDown(GLFW_KEY_D)) move += right;

    if (glm::length(move) > 0.001f)
        player.transform.translation += glm::normalize(move) * speed;

    glm::vec3 pos  = player.transform.translation;
    int       mode = player.playerCameraMode;
    if (mode == 1) {
        cam.pos   = pos + glm::vec3(0.0f, -5.0f, 3.0f);
        cam.front = glm::normalize(pos - cam.pos);
    } else if (mode == 2) {
        cam.pos   = pos + glm::vec3(0.0f, 0.0f, 15.0f);
        cam.front = glm::vec3(0.0f, 0.001f, -1.0f);
        cam.up    = glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Trigger zones
// ---------------------------------------------------------------------------

void Runtime::tickTriggers() {
    for (int i = 0; i < (int)scene->entities.size(); i++) {
        const auto& a = scene->entities[i];
        if (!a.isTrigger || !a.hasCollision) continue;

        glm::mat4 worldA = getWorldTransform(*scene, i);
        glm::vec3 posA   = glm::vec3(worldA[3]);
        float     radA   = std::max({glm::length(glm::vec3(worldA[0])),
                                     glm::length(glm::vec3(worldA[1])),
                                     glm::length(glm::vec3(worldA[2]))});

        for (int j = 0; j < (int)scene->entities.size(); j++) {
            if (i == j) continue;
            const auto& b = scene->entities[j];
            if (!b.hasCollision || b.isTrigger) continue;

            glm::mat4 worldB = getWorldTransform(*scene, j);
            glm::vec3 posB   = glm::vec3(worldB[3]);
            float     radB   = std::max({glm::length(glm::vec3(worldB[0])),
                                         glm::length(glm::vec3(worldB[1])),
                                         glm::length(glm::vec3(worldB[2]))});

            bool     overlapping = glm::length(posA - posB) < (radA + radB) * 0.5f;
            uint64_t key         = ((uint64_t)a.id << 32) | b.id;
            bool     was         = activeTriggerOverlaps.count(key) > 0;

            if (overlapping && !was) {
                activeTriggerOverlaps.insert(key);
                luaEngine.fireTriggerEnter(a.id, b.id);
            } else if (!overlapping && was) {
                activeTriggerOverlaps.erase(key);
                luaEngine.fireTriggerExit(a.id, b.id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interact system
// ---------------------------------------------------------------------------

void Runtime::tickInteract() {
    if (!cbIsKeyDown) return;
    bool isE   = cbIsKeyDown(GLFW_KEY_E);
    bool rising = isE && !wasInteractPressed;
    wasInteractPressed = isE;
    if (!rising) return;

    int playerIdx = -1;
    for (int i = 0; i < (int)scene->entities.size(); i++)
        if (scene->entities[i].isPlayer) { playerIdx = i; break; }
    if (playerIdx < 0) return;

    glm::vec3 playerPos = scene->entities[playerIdx].transform.translation;
    constexpr float kInteractRange = 3.0f;

    int   bestIdx  = -1;
    float bestDist = kInteractRange;
    for (int i = 0; i < (int)scene->entities.size(); i++) {
        if (i == playerIdx) continue;
        const auto& e = scene->entities[i];
        if (e.luaScriptFile.empty()) continue;
        glm::vec3 pos  = glm::vec3(getWorldTransform(*scene, i)[3]);
        float     dist = glm::length(pos - playerPos);
        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }
    if (bestIdx >= 0)
        luaEngine.fireInteract(scene->entities[bestIdx].id);
}

// ---------------------------------------------------------------------------
// AI / Pathfinding
// ---------------------------------------------------------------------------

static std::vector<glm::vec2> astar(const Navmesh& nav, glm::vec2 startW, glm::vec2 goalW) {
    int sx = (int)((startW.x - nav.originX) / nav.cellSize);
    int sy = (int)((startW.y - nav.originY) / nav.cellSize);
    int gx = (int)((goalW.x  - nav.originX) / nav.cellSize);
    int gy = (int)((goalW.y  - nav.originY) / nav.cellSize);

    sx = std::clamp(sx, 0, nav.width  - 1);
    sy = std::clamp(sy, 0, nav.depth  - 1);
    gx = std::clamp(gx, 0, nav.width  - 1);
    gy = std::clamp(gy, 0, nav.depth  - 1);

    if (!nav.cells[nav.idx(gx, gy)].walkable) return {};
    if (sx == gx && sy == gy) return { nav.worldPos(gx, gy) };

    const int N = nav.width * nav.depth;
    std::vector<float> g(N, 1e30f);
    std::vector<int>   parent(N, -1);
    std::vector<bool>  closed(N, false);

    using Pair = std::pair<float, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> open;

    int startIdx = nav.idx(sx, sy);
    int goalIdx  = nav.idx(gx, gy);
    g[startIdx]  = 0.0f;
    open.push({glm::length(glm::vec2(gx - sx, gy - sy)), startIdx});

    const int   dx[8]    = { 1,-1, 0, 0, 1, 1,-1,-1};
    const int   dy[8]    = { 0, 0, 1,-1, 1,-1, 1,-1};
    const float cost[8]  = { 1, 1, 1, 1, 1.414f, 1.414f, 1.414f, 1.414f};

    while (!open.empty()) {
        auto [fc, cur] = open.top(); open.pop();
        if (closed[cur]) continue;
        closed[cur] = true;
        if (cur == goalIdx) break;

        int cx = cur % nav.width;
        int cy = cur / nav.width;

        for (int d = 0; d < 8; d++) {
            int nx = cx + dx[d], ny = cy + dy[d];
            if (!nav.inBounds(nx, ny)) continue;
            int ni = nav.idx(nx, ny);
            if (closed[ni] || !nav.cells[ni].walkable) continue;
            float ng = g[cur] + cost[d];
            if (ng < g[ni]) {
                g[ni]      = ng;
                parent[ni] = cur;
                open.push({ng + glm::length(glm::vec2(gx - nx, gy - ny)), ni});
            }
        }
    }

    if (parent[goalIdx] < 0 && goalIdx != startIdx) return {};

    std::vector<glm::vec2> path;
    for (int cur = goalIdx; cur != startIdx && cur >= 0; cur = parent[cur])
        path.push_back(nav.worldPos(cur % nav.width, cur / nav.width));
    std::reverse(path.begin(), path.end());
    return path;
}

void Runtime::tickAI(float dt) {
    static constexpr float kRepathInterval = 1.5f;

    for (auto& entity : scene->entities) {
        if (!entity.hasAIAgent || entity.aiTargetEntityId == 0) continue;

        const GameObject* target = nullptr;
        for (const auto& e : scene->entities)
            if (e.id == entity.aiTargetEntityId) { target = &e; break; }
        if (!target) continue;

        AIAgentState& state = aiAgentStates[entity.id];
        state.repathTimer += dt;

        glm::vec3 agentPos  = entity.transform.translation;
        glm::vec3 targetPos = target->transform.translation;
        float     dist2D    = glm::length(glm::vec2(targetPos.x - agentPos.x, targetPos.y - agentPos.y));

        if (dist2D <= entity.aiStoppingDist) { state.path.clear(); continue; }

        bool targetMoved = glm::length(targetPos - state.lastTargetPos) > 1.5f;
        if (state.path.empty() || state.repathTimer >= kRepathInterval || targetMoved) {
            state.path = navmesh.valid()
                ? astar(navmesh, glm::vec2(agentPos.x, agentPos.y), glm::vec2(targetPos.x, targetPos.y))
                : std::vector<glm::vec2>{ glm::vec2(targetPos.x, targetPos.y) };
            state.lastTargetPos = targetPos;
            state.repathTimer   = 0.0f;
        }

        if (state.path.empty()) continue;

        glm::vec2 nextWP  = state.path.front();
        glm::vec2 agentXY = glm::vec2(agentPos.x, agentPos.y);
        glm::vec2 toWP    = nextWP - agentXY;
        float     toWPLen = glm::length(toWP);

        if (toWPLen < 0.1f) {
            state.path.erase(state.path.begin());
        } else {
            glm::vec2 dir  = toWP / toWPLen;
            float     step = std::min(entity.aiMoveSpeed * dt, toWPLen);
            entity.transform.translation.x += dir.x * step;
            entity.transform.translation.y += dir.y * step;
            entity.transform.rotation.z = atan2f(dir.y, dir.x) - glm::half_pi<float>();
        }
    }
}
