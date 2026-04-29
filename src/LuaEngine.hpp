#pragma once
extern "C" {
#include <lua5.4/lua.h>
#include <lua5.4/lualib.h>
#include <lua5.4/lauxlib.h>
}
#include <string>
#include <vector>
#include <map>
#include <functional>
#include "Scene.hpp"
#include "json.hpp"

struct DialogueState {
    bool        active         = false;
    std::string speaker;
    std::string text;
    std::vector<std::string> choices;  // empty = plain dismiss
    int         selectedChoice = -1;   // >= 0 when user picked a choice
    bool        dismissed      = false; // true when plain dialogue dismissed
};

struct LuaEntityState {
    int        tableRef    = LUA_NOREF;
    int        coRef       = LUA_NOREF;  // registry ref keeping coroutine alive
    lua_State* co          = nullptr;
    float      waitTimer   = 0.0f;       // engine.wait() countdown
    bool       waitDialogue= false;      // blocked on engine.showDialogue()
};

class LuaEngine {
public:
    void init(Scene* scene);
    void shutdown();

    // Called at play-start; scriptsDir = full path to project's scripts/ folder
    void onPlayStart(const std::string& scriptsDir);
    // Called at play-stop
    void onPlayStop();

    // Called every play frame
    void tick(float dt);

    // Script events fired externally
    void fireInteract(uint32_t entityId);
    void fireTriggerEnter(uint32_t triggerId, uint32_t otherId);
    void fireTriggerExit(uint32_t triggerId, uint32_t otherId);

    // Hot-reload: replace one entity's Lua state mid-play from a code string
    void hotReloadScript(uint32_t entityId, const std::string& code);

    // Public state read by VulkanEngine each frame
    DialogueState            dialogue;
    std::string              pendingSceneLoad;    // non-empty = engine.loadScene() was called
    std::vector<uint32_t>    pendingDestroyIds;   // entities to destroy this tick

    // Callbacks set by VulkanEngine before onPlayStart
    std::function<bool(int glfwKey)>              cbIsKeyDown;
    std::function<void(const std::string&)>       cbPlaySound;
    std::function<uint32_t(const std::string&)>   cbSpawnPrefab; // prefab name → new root id

    // Persistent save data (JSON key-value store, file at saveDataPath)
    nlohmann::json saveData;
    std::string    saveDataPath; // set by VulkanEngine before onPlayStart

    // Set just before any lua_resume so C callbacks know current entity
    uint32_t currentEntityId = 0;

private:
    lua_State* L     = nullptr;
    Scene*     scene = nullptr;

    std::map<uint32_t, LuaEntityState> states;

    void loadScript(uint32_t entityId, const std::string& fullPath);
    void resumeCoroutine(LuaEntityState& st, int nargs);
    bool startCoroutine(uint32_t entityId, LuaEntityState& st,
                        const char* funcName, std::vector<int> extraIntArgs = {});
    void finishCoroutine(LuaEntityState& st);
    void registerAPI();

    static int l_log(lua_State* L);
    static int l_showDialogue(lua_State* L);
    static int l_wait(lua_State* L);
    static int l_getPosition(lua_State* L);
    static int l_setPosition(lua_State* L);
    static int l_findByName(lua_State* L);
    static int l_getPlayer(lua_State* L);
    static int l_loadScene(lua_State* L);
    static int l_playSound(lua_State* L);
    static int l_isKeyDown(lua_State* L);
    static int l_getEntityName(lua_State* L);
    static int l_setUIText(lua_State* L);
    static int l_setUIValue(lua_State* L);
    static int l_getRotation(lua_State* L);
    static int l_setRotation(lua_State* L);
    static int l_getDistance(lua_State* L);
    static int l_moveToward(lua_State* L);
    static int l_setVisible(lua_State* L);
    static int l_saveData(lua_State* L);
    static int l_loadData(lua_State* L);
    static int l_destroyEntity(lua_State* L);
    static int l_spawnPrefab(lua_State* L);

    void flushSaveData();
};
