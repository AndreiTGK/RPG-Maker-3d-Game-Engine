#include "LuaEngine.hpp"
#include "EngineLog.hpp"
#include "PrefabSerializer.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <filesystem>

static const char* kRegEngine = "__lua_engine__";

static LuaEngine* getEng(lua_State* L) {
    return (LuaEngine*)lua_touserdata(L, lua_upvalueindex(1));
}

void LuaEngine::init(Scene* scene_) {
    scene = scene_;
}

void LuaEngine::shutdown() {
    onPlayStop();
}

void LuaEngine::onPlayStart(const std::string& scriptsDir) {
    onPlayStop();
    pendingDestroyIds.clear();

    // Load persistent save data
    saveData = nlohmann::json::object();
    if (!saveDataPath.empty()) {
        std::ifstream sf(saveDataPath);
        if (sf.is_open()) try { sf >> saveData; } catch (...) { saveData = nlohmann::json::object(); }
    }

    L = luaL_newstate();
    luaL_openlibs(L);

    lua_pushlightuserdata(L, this);
    lua_setfield(L, LUA_REGISTRYINDEX, kRegEngine);

    registerAPI();

    for (const auto& entity : scene->entities) {
        if (entity.luaScriptFile.empty()) continue;
        std::string fullPath = scriptsDir + "/" + entity.luaScriptFile;
        loadScript(entity.id, fullPath);
    }

    for (auto& [entityId, st] : states)
        startCoroutine(entityId, st, "onStart");
}

void LuaEngine::onPlayStop() {
    flushSaveData();
    if (L) { lua_close(L); L = nullptr; }
    states.clear();
    dialogue = {};
    pendingSceneLoad = "";
    pendingDestroyIds.clear();
    currentEntityId = 0;
}

void LuaEngine::flushSaveData() {
    if (saveDataPath.empty() || saveData.empty()) return;
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(saveDataPath).parent_path());
    std::ofstream sf(saveDataPath);
    if (sf.is_open()) sf << saveData.dump(4) << '\n';
}

void LuaEngine::tick(float dt) {
    if (!L) return;

    for (auto& [entityId, st] : states) {
        if (st.co != nullptr) {
            if (st.waitTimer > 0.0f) {
                st.waitTimer -= dt;
                if (st.waitTimer <= 0.0f) {
                    st.waitTimer = 0.0f;
                    currentEntityId = entityId;
                    resumeCoroutine(st, 0);
                }
            } else if (st.waitDialogue) {
                if (dialogue.dismissed) {
                    st.waitDialogue = false;
                    dialogue = {};
                    currentEntityId = entityId;
                    resumeCoroutine(st, 0);
                } else if (dialogue.selectedChoice >= 0) {
                    int choice = dialogue.selectedChoice;
                    st.waitDialogue = false;
                    dialogue = {};
                    currentEntityId = entityId;
                    lua_pushinteger(st.co, choice + 1); // 1-based
                    resumeCoroutine(st, 1);
                }
            }
        } else {
            // Non-blocking onUpdate
            lua_rawgeti(L, LUA_REGISTRYINDEX, st.tableRef);
            if (!lua_istable(L, -1)) { lua_pop(L, 1); continue; }
            lua_getfield(L, -1, "onUpdate");
            if (lua_isfunction(L, -1)) {
                lua_pushvalue(L, -2); // self
                lua_pushnumber(L, dt);
                if (lua_pcall(L, 2, 0, 0) != LUA_OK) {
                    LOG_ERROR("[Lua] onUpdate error: %s", lua_tostring(L, -1));
                    lua_pop(L, 1);
                }
            } else {
                lua_pop(L, 1);
            }
            lua_pop(L, 1);
        }
    }
}

void LuaEngine::fireInteract(uint32_t entityId) {
    if (!L) return;
    auto it = states.find(entityId);
    if (it == states.end()) return;
    auto& st = it->second;
    if (st.co != nullptr) return; // already in a blocking call
    currentEntityId = entityId;
    startCoroutine(entityId, st, "onInteract");
}

void LuaEngine::fireTriggerEnter(uint32_t triggerId, uint32_t otherId) {
    if (!L) return;
    auto it = states.find(triggerId);
    if (it == states.end()) return;
    auto& st = it->second;
    if (st.co != nullptr) return;
    currentEntityId = triggerId;
    startCoroutine(triggerId, st, "onTriggerEnter", {(int)otherId});
}

void LuaEngine::fireTriggerExit(uint32_t triggerId, uint32_t otherId) {
    if (!L) return;
    auto it = states.find(triggerId);
    if (it == states.end()) return;
    auto& st = it->second;
    if (st.co != nullptr) return;
    currentEntityId = triggerId;
    startCoroutine(triggerId, st, "onTriggerExit", {(int)otherId});
}

void LuaEngine::loadScript(uint32_t entityId, const std::string& fullPath) {
    std::ifstream f(fullPath);
    if (!f.is_open()) {
        LOG_ERROR("[Lua] Cannot open: %s", fullPath.c_str());
        return;
    }
    std::string code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    if (luaL_loadstring(L, code.c_str()) != LUA_OK) {
        LOG_ERROR("[Lua] Parse error in %s: %s", fullPath.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
        return;
    }
    if (lua_pcall(L, 0, 1, 0) != LUA_OK) {
        LOG_ERROR("[Lua] Runtime error in %s: %s", fullPath.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
        return;
    }
    if (!lua_istable(L, -1)) {
        LOG_ERROR("[Lua] Script must return a table: %s", fullPath.c_str());
        lua_pop(L, 1);
        return;
    }

    // Inject self.id so scripts can always know their own entity ID
    lua_pushinteger(L, (lua_Integer)entityId);
    lua_setfield(L, -2, "id");

    LuaEntityState st;
    st.tableRef = luaL_ref(L, LUA_REGISTRYINDEX);
    states[entityId] = st;
}

void LuaEngine::resumeCoroutine(LuaEntityState& st, int nargs) {
    int nresults = 0;
    int status = lua_resume(st.co, L, nargs, &nresults);

    if (status == LUA_OK) {
        if (nresults > 0) lua_pop(st.co, nresults);
        finishCoroutine(st);
    } else if (status == LUA_YIELD) {
        // still suspended, waitTimer or waitDialogue was set by the C callback
    } else {
        const char* msg = lua_tostring(st.co, -1);
        LOG_ERROR("[Lua] Coroutine error: %s", msg ? msg : "(no message)");
        lua_pop(st.co, 1);
        finishCoroutine(st);
    }
}

bool LuaEngine::startCoroutine(uint32_t entityId, LuaEntityState& st,
                                const char* funcName, std::vector<int> extraIntArgs) {
    lua_rawgeti(L, LUA_REGISTRYINDEX, st.tableRef);
    if (!lua_istable(L, -1)) { lua_pop(L, 1); return false; }

    lua_getfield(L, -1, funcName);
    if (!lua_isfunction(L, -1)) { lua_pop(L, 2); return false; }

    // Create thread and keep it alive in registry
    lua_State* co = lua_newthread(L);
    st.coRef = luaL_ref(L, LUA_REGISTRYINDEX); // pops thread from L
    st.co    = co;

    // Move function to coroutine
    lua_xmove(L, co, 1);

    // Push self table to coroutine
    lua_rawgeti(L, LUA_REGISTRYINDEX, st.tableRef);
    lua_xmove(L, co, 1);

    // Extra integer args
    for (int arg : extraIntArgs)
        lua_pushinteger(co, arg);

    lua_pop(L, 1); // pop table from L

    currentEntityId = entityId;
    resumeCoroutine(st, 1 + (int)extraIntArgs.size());
    return true;
}

void LuaEngine::finishCoroutine(LuaEntityState& st) {
    if (st.coRef != LUA_NOREF) {
        luaL_unref(L, LUA_REGISTRYINDEX, st.coRef);
        st.coRef = LUA_NOREF;
    }
    st.co          = nullptr;
    st.waitTimer   = 0.0f;
    st.waitDialogue= false;
}

void LuaEngine::registerAPI() {
    lua_newtable(L);

    auto reg = [&](const char* name, lua_CFunction fn) {
        lua_pushlightuserdata(L, this);
        lua_pushcclosure(L, fn, 1);
        lua_setfield(L, -2, name);
    };

    reg("log",          l_log);
    reg("showDialogue", l_showDialogue);
    reg("wait",         l_wait);
    reg("getPosition",  l_getPosition);
    reg("setPosition",  l_setPosition);
    reg("findByName",   l_findByName);
    reg("getPlayer",    l_getPlayer);
    reg("loadScene",    l_loadScene);
    reg("playSound",    l_playSound);
    reg("isKeyDown",    l_isKeyDown);
    reg("getEntityName",l_getEntityName);
    reg("setUIText",    l_setUIText);
    reg("setUIValue",   l_setUIValue);
    reg("getRotation",  l_getRotation);
    reg("setRotation",  l_setRotation);
    reg("getDistance",  l_getDistance);
    reg("moveToward",   l_moveToward);
    reg("setVisible",    l_setVisible);
    reg("saveData",      l_saveData);
    reg("loadData",      l_loadData);
    reg("destroyEntity", l_destroyEntity);
    reg("spawnPrefab",   l_spawnPrefab);

    lua_setglobal(L, "engine");

    // Expose GLFW key constants for engine.isKeyDown()
    lua_newtable(L);
    auto key = [&](const char* name, int val) {
        lua_pushinteger(L, val);
        lua_setfield(L, -2, name);
    };
    key("W", 87); key("A", 65); key("S", 83); key("D", 68);
    key("E", 69); key("Q", 81); key("SPACE", 32); key("LSHIFT", 340);
    key("UP", 265); key("DOWN", 264); key("LEFT", 263); key("RIGHT", 262);
    lua_setglobal(L, "Key");
}

// --- C API implementations ---

int LuaEngine::l_log(lua_State* L) {
    const char* msg = luaL_checkstring(L, 1);
    printf("[LuaScript] %s\n", msg);
    return 0;
}

int LuaEngine::l_showDialogue(lua_State* L) {
    LuaEngine* eng = getEng(L);
    const char* speaker = luaL_checkstring(L, 1);
    const char* text    = luaL_checkstring(L, 2);

    eng->dialogue = {};
    eng->dialogue.active  = true;
    eng->dialogue.speaker = speaker;
    eng->dialogue.text    = text;

    if (lua_istable(L, 3)) {
        int n = (int)lua_rawlen(L, 3);
        for (int i = 1; i <= n; i++) {
            lua_rawgeti(L, 3, i);
            eng->dialogue.choices.push_back(luaL_tolstring(L, -1, nullptr));
            lua_pop(L, 2); // luaL_tolstring pushes an extra string
        }
    }

    auto it = eng->states.find(eng->currentEntityId);
    if (it != eng->states.end())
        it->second.waitDialogue = true;

    return lua_yield(L, 0);
}

int LuaEngine::l_wait(lua_State* L) {
    LuaEngine* eng = getEng(L);
    float seconds = (float)luaL_checknumber(L, 1);

    auto it = eng->states.find(eng->currentEntityId);
    if (it != eng->states.end())
        it->second.waitTimer = seconds;

    return lua_yield(L, 0);
}

int LuaEngine::l_getPosition(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    for (const auto& e : eng->scene->entities) {
        if (e.id == id) {
            lua_pushnumber(L, e.transform.translation.x);
            lua_pushnumber(L, e.transform.translation.y);
            lua_pushnumber(L, e.transform.translation.z);
            return 3;
        }
    }
    return 0;
}

int LuaEngine::l_setPosition(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    float x = (float)luaL_checknumber(L, 2);
    float y = (float)luaL_checknumber(L, 3);
    float z = (float)luaL_checknumber(L, 4);
    for (auto& e : eng->scene->entities)
        if (e.id == id) { e.transform.translation = {x, y, z}; break; }
    return 0;
}

int LuaEngine::l_findByName(lua_State* L) {
    LuaEngine* eng = getEng(L);
    const char* name = luaL_checkstring(L, 1);
    for (const auto& e : eng->scene->entities) {
        if (e.name == name) { lua_pushinteger(L, e.id); return 1; }
    }
    lua_pushnil(L);
    return 1;
}

int LuaEngine::l_getPlayer(lua_State* L) {
    LuaEngine* eng = getEng(L);
    for (const auto& e : eng->scene->entities) {
        if (e.isPlayer) { lua_pushinteger(L, e.id); return 1; }
    }
    lua_pushnil(L);
    return 1;
}

int LuaEngine::l_loadScene(lua_State* L) {
    LuaEngine* eng = getEng(L);
    eng->pendingSceneLoad = luaL_checkstring(L, 1);
    return 0;
}

int LuaEngine::l_playSound(lua_State* L) {
    LuaEngine* eng = getEng(L);
    const char* filename = luaL_checkstring(L, 1);
    if (eng->cbPlaySound) eng->cbPlaySound(filename);
    return 0;
}

int LuaEngine::l_isKeyDown(lua_State* L) {
    LuaEngine* eng = getEng(L);
    int key = (int)luaL_checkinteger(L, 1);
    bool down = eng->cbIsKeyDown ? eng->cbIsKeyDown(key) : false;
    lua_pushboolean(L, down);
    return 1;
}

int LuaEngine::l_getEntityName(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    for (const auto& e : eng->scene->entities) {
        if (e.id == id) { lua_pushstring(L, e.name.c_str()); return 1; }
    }
    lua_pushnil(L);
    return 1;
}

int LuaEngine::l_setUIText(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id   = (uint32_t)luaL_checkinteger(L, 1);
    int      elem = (int)luaL_checkinteger(L, 2) - 1;
    const char* text = luaL_checkstring(L, 3);
    for (auto& e : eng->scene->entities) {
        if (e.id == id && e.hasUICanvas && elem >= 0 && elem < (int)e.uiCanvas.elements.size()) {
            e.uiCanvas.elements[elem].text = text;
            break;
        }
    }
    return 0;
}

int LuaEngine::l_setUIValue(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id   = (uint32_t)luaL_checkinteger(L, 1);
    int      elem = (int)luaL_checkinteger(L, 2) - 1;
    float    val  = (float)luaL_checknumber(L, 3);
    for (auto& e : eng->scene->entities) {
        if (e.id == id && e.hasUICanvas && elem >= 0 && elem < (int)e.uiCanvas.elements.size()) {
            e.uiCanvas.elements[elem].value = val;
            break;
        }
    }
    return 0;
}

int LuaEngine::l_getRotation(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    for (const auto& e : eng->scene->entities) {
        if (e.id == id) {
            lua_pushnumber(L, e.transform.rotation.x);
            lua_pushnumber(L, e.transform.rotation.y);
            lua_pushnumber(L, e.transform.rotation.z);
            return 3;
        }
    }
    return 0;
}

int LuaEngine::l_setRotation(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    float rx = (float)luaL_checknumber(L, 2);
    float ry = (float)luaL_checknumber(L, 3);
    float rz = (float)luaL_checknumber(L, 4);
    for (auto& e : eng->scene->entities)
        if (e.id == id) { e.transform.rotation = {rx, ry, rz}; break; }
    return 0;
}

int LuaEngine::l_getDistance(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t idA = (uint32_t)luaL_checkinteger(L, 1);
    uint32_t idB = (uint32_t)luaL_checkinteger(L, 2);
    glm::vec3 posA(0.0f), posB(0.0f);
    bool foundA = false, foundB = false;
    for (const auto& e : eng->scene->entities) {
        if (e.id == idA) { posA = e.transform.translation; foundA = true; }
        if (e.id == idB) { posB = e.transform.translation; foundB = true; }
        if (foundA && foundB) break;
    }
    lua_pushnumber(L, (foundA && foundB) ? (double)glm::length(posB - posA) : -1.0);
    return 1;
}

// engine.moveToward(id, tx, ty, tz, speed)  — moves entity toward target point by speed units
int LuaEngine::l_moveToward(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    float tx = (float)luaL_checknumber(L, 2);
    float ty = (float)luaL_checknumber(L, 3);
    float tz = (float)luaL_checknumber(L, 4);
    float spd = (float)luaL_checknumber(L, 5);
    for (auto& e : eng->scene->entities) {
        if (e.id == id) {
            glm::vec3 dir = glm::vec3(tx, ty, tz) - e.transform.translation;
            float dist = glm::length(dir);
            if (dist > 0.001f)
                e.transform.translation += glm::normalize(dir) * glm::min(spd, dist);
            break;
        }
    }
    return 0;
}

void LuaEngine::hotReloadScript(uint32_t entityId, const std::string& code) {
    if (!L) return;

    // Tear down existing state for this entity
    auto it = states.find(entityId);
    if (it != states.end()) {
        finishCoroutine(it->second);
        if (it->second.tableRef != LUA_NOREF)
            luaL_unref(L, LUA_REGISTRYINDEX, it->second.tableRef);
        states.erase(it);
    }

    if (luaL_loadstring(L, code.c_str()) != LUA_OK) {
        LOG_ERROR("[Lua] HotReload parse error (entity %u): %s", entityId, lua_tostring(L, -1));
        lua_pop(L, 1);
        return;
    }
    if (lua_pcall(L, 0, 1, 0) != LUA_OK) {
        LOG_ERROR("[Lua] HotReload runtime error (entity %u): %s", entityId, lua_tostring(L, -1));
        lua_pop(L, 1);
        return;
    }
    if (!lua_istable(L, -1)) {
        LOG_ERROR("[Lua] HotReload: script must return a table (entity %u)", entityId);
        lua_pop(L, 1);
        return;
    }

    lua_pushinteger(L, (lua_Integer)entityId);
    lua_setfield(L, -2, "id");

    LuaEntityState st;
    st.tableRef = luaL_ref(L, LUA_REGISTRYINDEX);
    states[entityId] = st;
    startCoroutine(entityId, states[entityId], "onStart");
}

int LuaEngine::l_setVisible(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    bool vis = lua_toboolean(L, 2) != 0;
    for (auto& e : eng->scene->entities)
        if (e.id == id) { e.visible = vis; break; }
    return 0;
}

// engine.saveData(key, value)  — persists a string/number/bool to save file
int LuaEngine::l_saveData(lua_State* L) {
    LuaEngine* eng = getEng(L);
    const char* key = luaL_checkstring(L, 1);
    if (lua_type(L, 2) == LUA_TBOOLEAN)
        eng->saveData[key] = lua_toboolean(L, 2) != 0;
    else if (lua_type(L, 2) == LUA_TNUMBER)
        eng->saveData[key] = lua_tonumber(L, 2);
    else
        eng->saveData[key] = luaL_checkstring(L, 2);
    eng->flushSaveData();
    return 0;
}

// engine.loadData(key)  — returns saved value or nil
int LuaEngine::l_loadData(lua_State* L) {
    LuaEngine* eng = getEng(L);
    const char* key = luaL_checkstring(L, 1);
    if (!eng->saveData.contains(key)) { lua_pushnil(L); return 1; }
    const auto& v = eng->saveData[key];
    if (v.is_boolean())      lua_pushboolean(L, v.get<bool>() ? 1 : 0);
    else if (v.is_number())  lua_pushnumber(L, v.get<double>());
    else if (v.is_string())  lua_pushstring(L, v.get<std::string>().c_str());
    else                     lua_pushnil(L);
    return 1;
}

// engine.destroyEntity(id)  — queues entity for deletion at end of tick
int LuaEngine::l_destroyEntity(lua_State* L) {
    LuaEngine* eng = getEng(L);
    uint32_t id = (uint32_t)luaL_checkinteger(L, 1);
    eng->pendingDestroyIds.push_back(id);
    return 0;
}

// engine.spawnPrefab(name)  — instantiate prefab, return new root entity id (or 0)
int LuaEngine::l_spawnPrefab(lua_State* L) {
    LuaEngine* eng = getEng(L);
    const char* name = luaL_checkstring(L, 1);
    if (eng->cbSpawnPrefab) {
        uint32_t id = eng->cbSpawnPrefab(name);
        lua_pushinteger(L, id);
    } else {
        lua_pushinteger(L, 0);
    }
    return 1;
}
