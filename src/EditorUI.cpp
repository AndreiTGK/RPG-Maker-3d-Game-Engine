#include "EditorUI.hpp"
#include "EngineLog.hpp"
#include "PrefabSerializer.hpp"
#include "imgui/imgui.h"
#include "ImGuizmo.h"
#include <glm/gtc/type_ptr.hpp>
#include <filesystem>
#include <set>
#include <algorithm>
#include <string>

// ─── Accent colours for section headers ──────────────────────────────────────
static constexpr ImVec4 kColTransform = {0.16f, 0.32f, 0.58f, 0.90f};
static constexpr ImVec4 kColMesh      = {0.14f, 0.44f, 0.48f, 0.90f};
static constexpr ImVec4 kColPhysics   = {0.46f, 0.28f, 0.08f, 0.90f};
static constexpr ImVec4 kColLight     = {0.48f, 0.40f, 0.06f, 0.90f};
static constexpr ImVec4 kColTerrain   = {0.14f, 0.40f, 0.14f, 0.90f};
static constexpr ImVec4 kColAudio     = {0.36f, 0.14f, 0.48f, 0.90f};
static constexpr ImVec4 kColParticles = {0.48f, 0.12f, 0.34f, 0.90f};
static constexpr ImVec4 kColAnim      = {0.12f, 0.38f, 0.44f, 0.90f};
static constexpr ImVec4 kColAI        = {0.46f, 0.12f, 0.12f, 0.90f};
static constexpr ImVec4 kColCanvas    = {0.34f, 0.12f, 0.48f, 0.90f};
static constexpr ImVec4 kColScript    = {0.14f, 0.44f, 0.20f, 0.90f};

static void pushHdr(ImVec4 b) {
    auto li = [](ImVec4 c, float d) {
        return ImVec4(std::min(c.x+d,1.f), std::min(c.y+d,1.f), std::min(c.z+d,1.f), c.w);
    };
    ImGui::PushStyleColor(ImGuiCol_Header,        b);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, li(b, 0.09f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  li(b, 0.16f));
}
static void popHdr() { ImGui::PopStyleColor(3); }

// Checkbox + coloured CollapsingHeader in one line.
// Writes enabled back, returns true only when enabled AND header is open.
static bool componentHeader(const char* label, bool& enabled, ImVec4 col, bool& dirty) {
    ImGui::PushID(label);
    if (ImGui::Checkbox("##en", &enabled)) dirty = true;
    ImGui::PopID();
    ImGui::SameLine();
    pushHdr(col);
    bool open = ImGui::CollapsingHeader(label);
    popHdr();
    return enabled && open;
}

// ─── Layout ──────────────────────────────────────────────────────────────────
struct Layout {
    float W, H;
    float topH, botH, leftW, rightW;
    float midX() const { return leftW; }
    float midY() const { return topH; }
    float midW() const { return W - leftW - rightW; }
    float midH() const { return H - topH - botH; }
};
static Layout computeLayout(float W, float H) {
    Layout L;
    L.W     = W;  L.H    = H;
    L.topH  = 38.0f;
    L.leftW = std::clamp(W * 0.195f, 210.0f, 310.0f);
    L.rightW= std::clamp(W * 0.220f, 250.0f, 355.0f);
    L.botH  = std::clamp(H * 0.250f, 155.0f, 260.0f);
    return L;
}

static constexpr ImGuiWindowFlags kDock =
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoCollapse |
    ImGuiWindowFlags_NoFocusOnAppearing;

// ─── Entity tree ─────────────────────────────────────────────────────────────
static void renderEntityTree(Scene* scene, int idx, int& sel,
                             bool& dirty, uint32_t& pendingDel, uint32_t& pendingDup,
                             uint32_t& pendingSavePrefab) {
    if (idx < 0 || idx >= (int)scene->entities.size()) return;
    const GameObject& obj = scene->entities[idx];
    bool hasChildren = false;
    for (const auto& e : scene->entities)
        if (e.parentId == obj.id) { hasChildren = true; break; }

    ImGuiTreeNodeFlags f = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (!hasChildren) f |= ImGuiTreeNodeFlags_Leaf;
    if (sel == idx)   f |= ImGuiTreeNodeFlags_Selected;

    bool open = ImGui::TreeNodeEx((void*)(intptr_t)(uintptr_t)obj.id, f, "%s", obj.name.c_str());
    if (ImGui::IsItemClicked()) sel = idx;

    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Duplicate")) { pendingDup = obj.id; sel = idx; dirty = true; }
        if (ImGui::MenuItem("Save as Prefab")) { pendingSavePrefab = obj.id; sel = idx; }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete")) { pendingDel = obj.id; dirty = true; }
        ImGui::EndPopup();
    }
    if (open) {
        for (int i = 0; i < (int)scene->entities.size(); i++)
            if (scene->entities[i].parentId == obj.id)
                renderEntityTree(scene, i, sel, dirty, pendingDel, pendingDup, pendingSavePrefab);
        ImGui::TreePop();
    }
}

// ─── Inspector context (passed to section helpers) ───────────────────────────
struct InspCtx {
    Scene*                    scene;
    std::vector<std::string>* models;
    std::vector<std::string>* textures;
    std::vector<std::string>* audio;
    int*                      selModel;
    int*                      selTexture;
    EditorCallbacks&          cb;
    bool&                     dirty;
};

// ─── Inspector sections ───────────────────────────────────────────────────────

static void inspTransform(GameObject& obj, EditorCallbacks& cb, bool& dirty) {
    pushHdr(kColTransform);
    bool open = ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen);
    popHdr();
    if (!open) return;

    if (ImGui::Checkbox("Visible", &obj.visible)) dirty = true;
    ImGui::Separator();
    if (ImGui::DragFloat3("Position##t", &obj.transform.translation.x, 0.1f)) dirty = true;
    if (ImGui::IsItemActivated()) cb.requestUndoSnapshot();
    if (ImGui::DragFloat3("Rotation##t", &obj.transform.rotation.x, 0.05f))   dirty = true;
    if (ImGui::IsItemActivated()) cb.requestUndoSnapshot();
    if (ImGui::DragFloat3("Scale##t",    &obj.transform.scale.x,     0.1f))   dirty = true;
    if (ImGui::IsItemActivated()) cb.requestUndoSnapshot();
}

static void inspMeshMaterial(GameObject& obj, InspCtx& ctx) {
    pushHdr(kColMesh);
    bool open = ImGui::CollapsingHeader("Mesh & Material", ImGuiTreeNodeFlags_DefaultOpen);
    popHdr();
    if (!open) return;

    if (ctx.models && !ctx.models->empty()) {
        std::vector<const char*> ns;
        for (const auto& m : *ctx.models) ns.push_back(m.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##model", ctx.selModel, ns.data(), (int)ns.size())) ctx.dirty = true;
        if (ImGui::Button("Apply Model", {-1, 0})) { ctx.cb.applyModel(); ctx.dirty = true; }
    } else {
        ImGui::TextColored({1.0f,0.5f,0.1f,1.0f}, "No models in project");
    }
    ImGui::Spacing();
    if (ctx.textures && !ctx.textures->empty()) {
        std::vector<const char*> ns;
        for (const auto& t : *ctx.textures) ns.push_back(t.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##tex", ctx.selTexture, ns.data(), (int)ns.size())) ctx.dirty = true;
        if (ImGui::Button("Apply Texture", {-1, 0})) {
            obj.textureIndex = *ctx.selTexture; ctx.dirty = true;
        }
    } else {
        ImGui::TextColored({1.0f,0.5f,0.1f,1.0f}, "No textures in project");
    }
    ImGui::Spacing();
    if (ImGui::SliderFloat("Metallic",  &obj.metallic,  0.0f, 1.0f)) ctx.dirty = true;
    if (ImGui::IsItemActivated()) ctx.cb.requestUndoSnapshot();
    if (ImGui::SliderFloat("Roughness", &obj.roughness, 0.0f, 1.0f)) ctx.dirty = true;
    if (ImGui::IsItemActivated()) ctx.cb.requestUndoSnapshot();
}

static void inspPhysics(GameObject& obj, bool& dirty) {
    pushHdr(kColPhysics);
    bool open = ImGui::CollapsingHeader("Physics");
    popHdr();
    if (!open) return;

    if (ImGui::Checkbox("Collision",  &obj.hasCollision))  dirty = true;
    if (ImGui::Checkbox("Static",     &obj.isStatic))      dirty = true;
    if (!obj.isStatic)
        if (ImGui::Checkbox("Gravity",&obj.gravityEnabled)) dirty = true;
}

static void inspLight(GameObject& obj, Scene* scene, bool& dirty) {
    if (!componentHeader("Light Source", obj.isLightSource, kColLight, dirty)) return;

    int cnt = 0;
    for (const auto& e : scene->entities) if (e.isLightSource) cnt++;
    if (cnt > 4)
        ImGui::TextColored({1.0f,0.4f,0.0f,1.0f}, "Warning: %d lights active, only 4 used", cnt);

    if (ImGui::DragFloat("Intensity", &obj.lightIntensity, 0.1f, 0.0f, 200.0f)) dirty = true;
    if (ImGui::ColorEdit3("Color",    &obj.lightColor.x))                        dirty = true;
    if (ImGui::DragFloat("Radius",    &obj.lightRadius,    0.5f, 0.1f, 500.0f)) dirty = true;
    const char* types[] = {"Point", "Spot"};
    int t = (int)obj.lightType;
    if (ImGui::Combo("Type", &t, types, 2)) { obj.lightType = (LightType)t; dirty = true; }
    if (obj.lightType == LightType::Spot) {
        if (ImGui::SliderFloat("Spot Angle", &obj.spotAngle, 1.0f, 89.0f)) dirty = true;
        ImGui::TextDisabled("Spot not yet in shaders");
    }
}

static void inspTerrain(GameObject& obj, bool& dirty) {
    if (!obj.isTerrain) return;
    pushHdr(kColTerrain);
    bool open = ImGui::CollapsingHeader("Terrain");
    popHdr();
    if (!open) return;

    bool ch = false;
    ch |= ImGui::DragInt  ("Width (quads)",  &obj.terrainWidth,     1, 2, 512);
    ch |= ImGui::DragInt  ("Depth (quads)",  &obj.terrainDepth,     1, 2, 512);
    ch |= ImGui::DragFloat("Amplitude",      &obj.terrainAmplitude, 0.1f, 0.1f, 100.0f);
    ch |= ImGui::DragFloat("Frequency",      &obj.terrainFrequency, 0.001f, 0.001f, 1.0f);
    ch |= ImGui::DragInt  ("Seed",           &obj.terrainSeed, 1);
    if (ch) { obj.terrainDirty = true; dirty = true; }
    if (ImGui::Button("Regenerate", {-1, 0})) { obj.terrainDirty = true; dirty = true; }
}

static void inspAudio(GameObject& obj, std::vector<std::string>* audioList, bool& dirty) {
    if (!componentHeader("Audio Source", obj.hasAudio, kColAudio, dirty)) return;

    if (audioList && !audioList->empty()) {
        int idx = 0;
        for (int i = 0; i < (int)audioList->size(); i++)
            if ((*audioList)[i] == obj.audioFile) { idx = i; break; }
        std::vector<const char*> items;
        for (const auto& a : *audioList) items.push_back(a.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##audio", &idx, items.data(), (int)items.size())) {
            obj.audioFile = (*audioList)[idx]; dirty = true;
        }
    } else {
        ImGui::TextDisabled("No files in assets/audio/");
    }
    if (ImGui::Checkbox("Loop",   &obj.audioLoop))                           dirty = true;
    if (ImGui::SliderFloat("Volume",   &obj.audioVolume,  0.0f, 1.0f))       dirty = true;
    if (ImGui::DragFloat("Min Dist",   &obj.audioMinDist, 0.1f, 0.1f, 100.0f)) dirty = true;
    if (ImGui::DragFloat("Max Dist",   &obj.audioMaxDist, 0.1f, 0.1f, 500.0f)) dirty = true;
}

static void inspParticles(GameObject& obj, bool& dirty) {
    if (!componentHeader("Particle Emitter", obj.hasEmitter, kColParticles, dirty)) return;

    if (ImGui::DragInt  ("Max Particles", &obj.emitterMaxParticles, 1, 1, 5000))        dirty = true;
    if (ImGui::DragFloat("Emit Rate",     &obj.emitterRate,    0.5f, 0.1f, 1000.0f))   dirty = true;
    if (ImGui::DragFloat("Lifetime",      &obj.emitterLifetime,0.1f, 0.05f, 30.0f))    dirty = true;
    if (ImGui::DragFloat("Speed",         &obj.emitterSpeed,   0.1f, 0.0f, 100.0f))    dirty = true;
    if (ImGui::SliderFloat("Spread",      &obj.emitterSpread,  0.0f, 3.14159f))        dirty = true;
    if (ImGui::DragFloat("Start Size",    &obj.emitterStartSize,0.01f, 0.0f, 10.0f))   dirty = true;
    if (ImGui::DragFloat("End Size",      &obj.emitterEndSize,  0.01f, 0.0f, 10.0f))   dirty = true;
    if (ImGui::ColorEdit4("Start Color",  &obj.emitterStartColor.r))                    dirty = true;
    if (ImGui::ColorEdit4("End Color",    &obj.emitterEndColor.r))                      dirty = true;
}

static void inspAnimation(GameObject& obj, std::vector<std::string>* models,
                           EditorCallbacks& cb, bool& dirty) {
    if (!componentHeader("Skeletal Animation", obj.hasSkin, kColAnim, dirty)) return;

    if (models && !models->empty()) {
        const char* cur = obj.skinModelName.empty() ? "(Choose GLTF model)" : obj.skinModelName.c_str();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##skinmdl", cur)) {
            for (const auto& m : *models) {
                bool gltf = (m.size()>=5 && m.substr(m.size()-5)==".gltf") ||
                            (m.size()>=4 && m.substr(m.size()-4)==".glb");
                if (!gltf) continue;
                bool sel = (obj.skinModelName == m);
                if (ImGui::Selectable(m.c_str(), sel)) {
                    obj.skinModelName = m; obj.currentAnimation = 0; dirty = true;
                }
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }
    if (!obj.skinModelName.empty() && cb.getAnimationNames) {
        auto names = cb.getAnimationNames(obj.skinModelName);
        if (!names.empty()) {
            int idx = std::clamp(obj.currentAnimation, 0, (int)names.size()-1);
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##anim", names[idx].c_str())) {
                for (int i = 0; i < (int)names.size(); i++) {
                    if (ImGui::Selectable(names[i].c_str(), idx==i)) {
                        obj.currentAnimation = i; dirty = true;
                    }
                    if (idx==i) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
    }
    if (ImGui::Button(obj.animationPlaying ? "Pause" : "Play")) {
        obj.animationPlaying = !obj.animationPlaying; dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Loop", &obj.animationLoop)) dirty = true;
    if (ImGui::DragFloat("Speed", &obj.animationSpeed, 0.01f, 0.01f, 10.0f)) dirty = true;
}

static void inspAI(GameObject& obj, Scene* scene, bool& dirty) {
    if (!componentHeader("AI Agent", obj.hasAIAgent, kColAI, dirty)) return;

    const char* tgt = "(None)";
    for (const auto& e : scene->entities)
        if (e.id == obj.aiTargetEntityId) { tgt = e.name.c_str(); break; }
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##aitgt", obj.aiTargetEntityId ? tgt : "(None)")) {
        if (ImGui::Selectable("(None)", obj.aiTargetEntityId==0)) { obj.aiTargetEntityId=0; dirty=true; }
        for (const auto& e : scene->entities) {
            if (e.id == obj.id) continue;
            bool sel = (obj.aiTargetEntityId == e.id);
            if (ImGui::Selectable(e.name.c_str(), sel)) { obj.aiTargetEntityId=e.id; dirty=true; }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::TextDisabled("Target Entity");
    if (ImGui::DragFloat("Move Speed",    &obj.aiMoveSpeed,    0.1f, 0.1f, 50.0f))   dirty = true;
    if (ImGui::DragFloat("Stop Distance", &obj.aiStoppingDist, 0.05f,0.05f,20.0f))   dirty = true;
}

static void inspUICanvas(GameObject& obj, bool& dirty) {
    if (!componentHeader("UI Canvas", obj.hasUICanvas, kColCanvas, dirty)) return;

    if (ImGui::Checkbox("Visible", &obj.uiCanvas.visible)) dirty = true;
    ImGui::Separator();

    static const char* kTypes[]   = {"Label","Button","Image","Healthbar"};
    static const char* kAnchors[] = {"Top-Left","Top-Center","Top-Right",
                                      "Mid-Left","Center","Mid-Right",
                                      "Bot-Left","Bot-Center","Bot-Right"};
    for (int ei = 0; ei < (int)obj.uiCanvas.elements.size(); ei++) {
        UIElement& el = obj.uiCanvas.elements[ei];
        ImGui::PushID(ei);
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "[%d] %s", ei, kTypes[(int)el.type]);
        bool open = ImGui::TreeNodeEx(hdr, ImGuiTreeNodeFlags_DefaultOpen);
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) {
            obj.uiCanvas.elements.erase(obj.uiCanvas.elements.begin() + ei);
            dirty = true; ImGui::PopID();
            if (open) ImGui::TreePop();
            break;
        }
        if (open) {
            int t = (int)el.type;
            if (ImGui::Combo("Type",   &t, kTypes,   4)) { el.type  =(UIElementType)t; dirty=true; }
            int a = (int)el.anchor;
            if (ImGui::Combo("Anchor", &a, kAnchors, 9)) { el.anchor=(UIAnchor)a;      dirty=true; }
            if (ImGui::DragFloat2("Offset px", &el.offset.x, 1.0f))        dirty=true;
            if (ImGui::DragFloat2("Size px",   &el.size.x,   1.0f, 1.0f))  dirty=true;
            if (el.type != UIElementType::Image) {
                static char tb[256];
                strncpy(tb, el.text.c_str(), sizeof(tb)-1);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputText("##eltext", tb, sizeof(tb))) { el.text=tb; dirty=true; }
                ImGui::TextDisabled("Text");
            }
            if (ImGui::ColorEdit4("Text/Tint", &el.color.x))   dirty=true;
            if (ImGui::ColorEdit4("Background", &el.bgColor.x)) dirty=true;
            if (el.type == UIElementType::Healthbar) {
                if (ImGui::SliderFloat("Value",      &el.value,      0.0f,1.0f)) dirty=true;
                if (ImGui::ColorEdit4("Fill Color",  &el.valueColor.x))          dirty=true;
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (ImGui::Button("+ Add Element", {-1,0})) { obj.uiCanvas.elements.push_back({}); dirty=true; }
}

// ─── Visual script compiler ───────────────────────────────────────────────────

static std::string vsCompileAction(const VSAction& a) {
    const auto& p = a.p;
    auto s  = [&](int i) -> std::string {
        return (i < 5 && !p[i].empty()) ? p[i] : "";
    };
    auto str = [&](int i, const std::string& def = "") -> std::string {
        std::string v = s(i); if (v.empty()) v = def;
        return "\"" + v + "\"";
    };
    auto num = [&](int i, const std::string& def = "0") -> std::string {
        std::string v = s(i); return v.empty() ? def : v;
    };
    auto ent = [&](int i) -> std::string {
        std::string v = s(i);
        return (v.empty() || v == "self") ? "self.id" : v;
    };
    switch (a.type) {
    case VSActionType::Log:
        return "  engine.log(" + str(0,"") + ")\n";
    case VSActionType::PlaySound:
        return "  engine.playSound(" + str(0,"") + ")\n";
    case VSActionType::SetPosition:
        return "  engine.setPosition(" + ent(0) + ", " + num(1) + ", " + num(2) + ", " + num(3) + ")\n";
    case VSActionType::MoveToward:
        return "  engine.moveToward(" + ent(0) + ", " + num(1) + ", " + num(2) + ", " + num(3) + ", " + num(4,"1") + ")\n";
    case VSActionType::SetVisible:
        return "  engine.setVisible(" + ent(0) + ", " + (s(1)=="false"?"false":"true") + ")\n";
    case VSActionType::SetRotation:
        return "  engine.setRotation(" + ent(0) + ", " + num(1) + ", " + num(2) + ", " + num(3) + ")\n";
    case VSActionType::LoadScene:
        return "  engine.loadScene(" + str(0,"") + ")\n";
    case VSActionType::ShowDialogue:
        return "  engine.showDialogue(" + str(0,"NPC") + ", " + str(1,"Hello!") + ")\n";
    case VSActionType::Wait:
        return "  engine.wait(" + num(0,"1") + ")\n";
    case VSActionType::SetUIText:
        return "  engine.setUIText(" + ent(0) + ", " + num(1,"1") + ", " + str(2,"") + ")\n";
    case VSActionType::SetUIValue:
        return "  engine.setUIValue(" + ent(0) + ", " + num(1,"1") + ", " + num(2,"1") + ")\n";
    }
    return "";
}

static std::string compileVisualScript(const VisualScript& vs) {
    static const char* kFuncs[] = {
        "onStart(self)", "onUpdate(self, dt)", "onInteract(self)",
        "onTriggerEnter(self, otherId)", "onTriggerExit(self, otherId)"
    };
    std::string code = "local M = {}\n\n";
    for (int ev = 0; ev < 5; ev++) {
        bool any = false;
        for (const auto& blk : vs.blocks) if ((int)blk.event == ev) { any = true; break; }
        if (!any) continue;
        code += "function M."; code += kFuncs[ev]; code += "\n";
        for (const auto& blk : vs.blocks)
            if ((int)blk.event == ev)
                for (const auto& act : blk.actions)
                    code += vsCompileAction(act);
        code += "end\n\n";
    }
    code += "return M\n";
    return code;
}

// Renders the visual script block editor. Returns true if anything changed.
static bool renderVisualScript(VisualScript& vs, int objIdx) {
    static const char* kEvents[]  = {"OnStart","OnUpdate","OnInteract","OnTriggerEnter","OnTriggerExit"};
    static const char* kActions[] = {"Log","PlaySound","SetPosition","MoveToward",
                                     "SetVisible","SetRotation","LoadScene",
                                     "ShowDialogue","Wait","SetUIText","SetUIValue"};
    static const char* kActionLabels[][5] = {
        {"Message","","","",""},                // Log
        {"File","","","",""},                   // PlaySound
        {"Entity","X","Y","Z",""},              // SetPosition
        {"Entity","X","Y","Z","Speed"},         // MoveToward
        {"Entity","Visible","","",""},          // SetVisible (true/false)
        {"Entity","RX","RY","RZ",""},           // SetRotation
        {"Scene","","","",""},                  // LoadScene
        {"Speaker","Text","","",""},            // ShowDialogue
        {"Seconds","","","",""},                // Wait
        {"Entity","Elem#","Text","",""},        // SetUIText
        {"Entity","Elem#","Value","",""},       // SetUIValue
    };
    static const int kActionParamCount[] = {1,1,4,5,2,4,1,2,1,3,3};

    bool changed = false;
    char id[128];

    int blockToRemove = -1;
    for (int bi = 0; bi < (int)vs.blocks.size(); bi++) {
        auto& blk = vs.blocks[bi];
        ImGui::PushID(bi);

        // Block header row
        ImGui::SetNextItemWidth(120.0f);
        int ev = (int)blk.event;
        if (ImGui::Combo("##evt", &ev, kEvents, 5)) { blk.event = (VSEventType)ev; changed = true; }
        ImGui::SameLine();
        ImGui::TextDisabled("block");
        ImGui::SameLine(0, 8);
        if (ImGui::SmallButton("x##blk")) { blockToRemove = bi; changed = true; }

        // Actions
        int actToRemove = -1;
        for (int ai = 0; ai < (int)blk.actions.size(); ai++) {
            auto& act = blk.actions[ai];
            ImGui::PushID(ai);

            ImGui::Indent(12.0f);
            ImGui::SetNextItemWidth(100.0f);
            int at = (int)act.type;
            if (ImGui::Combo("##at", &at, kActions, 11)) { act.type = (VSActionType)at; changed = true; }
            ImGui::SameLine();
            if (ImGui::SmallButton("x##act")) { actToRemove = ai; changed = true; }

            int nparams = kActionParamCount[at];
            for (int pi = 0; pi < nparams; pi++) {
                snprintf(id, sizeof(id), "##p%d_%d_%d_%d", pi, objIdx, bi, ai);
                ImGui::SetNextItemWidth(-1.0f);
                char buf[256] = {};
                strncpy(buf, act.p[pi].c_str(), sizeof(buf)-1);
                ImGui::TextDisabled("%s", kActionLabels[at][pi]);
                ImGui::SameLine(70.0f);
                ImGui::SetNextItemWidth(-1.0f);
                if (ImGui::InputText(id, buf, sizeof(buf)))
                    { act.p[pi] = buf; changed = true; }
            }
            ImGui::Unindent(12.0f);
            ImGui::PopID();
        }
        if (actToRemove >= 0) blk.actions.erase(blk.actions.begin() + actToRemove);

        if (ImGui::SmallButton("+ Action")) { blk.actions.push_back({}); changed = true; }
        ImGui::Separator();
        ImGui::PopID();
    }
    if (blockToRemove >= 0) vs.blocks.erase(vs.blocks.begin() + blockToRemove);

    if (ImGui::Button("+ Block", {-1,0})) {
        VSBlock blk; blk.actions.push_back({});
        vs.blocks.push_back(blk);
        changed = true;
    }
    return changed;
}

static void inspScripting(GameObject& obj, EditorCallbacks& cb, bool& dirty,
                           bool& showEditor, std::string& edContent,
                           std::string& edFile, bool& edDirty,
                           uint32_t& edEntityId, int /*objIdx*/,
                           bool& showVsEditor, uint32_t& vsEdEntityId) {
    pushHdr(kColScript);
    bool open = ImGui::CollapsingHeader("Scripting");
    popHdr();
    if (!open) return;

    if (ImGui::Checkbox("Trigger Zone", &obj.isTrigger)) dirty = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Detects overlaps without physics response.\nFires onTriggerEnter/Exit in Lua.");

    if (ImGui::Checkbox("Is Player", &obj.isPlayer)) dirty = true;
    if (obj.isPlayer) {
        static const char* modes[] = {"FreeFly","Third Person","Top Down"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##cammode", &obj.playerCameraMode, modes, 3)) dirty = true;
        ImGui::TextDisabled("Camera Mode");
        if (ImGui::DragFloat("Move Speed", &obj.playerMoveSpeed, 0.1f, 0.1f, 50.0f)) dirty = true;
    }
    ImGui::Separator();

    static char fileBuf[256] = {};
    strncpy(fileBuf, obj.luaScriptFile.c_str(), sizeof(fileBuf)-1);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##scriptfile", "script.lua", fileBuf, sizeof(fileBuf))) {
        obj.luaScriptFile = fileBuf; dirty = true;
    }
    ImGui::TextDisabled("Script File (.lua)");
    if (ImGui::Button("Edit Script", {-1,0}) && cb.readScriptFile) {
        if (obj.luaScriptFile.empty()) { obj.luaScriptFile = obj.name + ".lua"; dirty = true; }
        std::string content = cb.readScriptFile(obj.luaScriptFile);
        if (content.empty()) {
            content = "local M = {}\n\n"
                      "function M.onStart(self)\nend\n\n"
                      "function M.onUpdate(self, dt)\nend\n\n"
                      "function M.onInteract(self)\n"
                      "    engine.showDialogue(\"NPC\", \"Hello!\")\nend\n\n"
                      "function M.onTriggerEnter(self, otherId)\nend\n\n"
                      "function M.onTriggerExit(self, otherId)\nend\n\nreturn M\n";
            if (cb.writeScriptFile) cb.writeScriptFile(obj.luaScriptFile, content);
        }
        edContent = content; edFile = obj.luaScriptFile; edDirty = false;
        edEntityId = obj.id; showEditor = true;
    }

    ImGui::Separator();
    // ── Visual Script ────────────────────────────────────────────────────────
    bool vsOpen = ImGui::TreeNodeEx("Visual Script##vs", ImGuiTreeNodeFlags_DefaultOpen);
    if (vsOpen) {
        bool vsEnable = obj.visualScript.enabled;
        if (ImGui::Checkbox("Enable##vse", &vsEnable)) {
            obj.visualScript.enabled = vsEnable;
            dirty = true;
        }
        if (obj.visualScript.enabled) {
            if (ImGui::Button("Open Visual Editor", {-1, 0})) {
                showVsEditor   = true;
                vsEdEntityId   = obj.id;
            }
            int nb = (int)obj.visualScript.blocks.size();
            ImGui::TextDisabled("%d block%s", nb, nb==1?"":"s");
        }
        ImGui::TreePop();
    }
}

// ─── Theme ───────────────────────────────────────────────────────────────────
void EditorUI::applyTheme() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 0.0f;
    s.ChildRounding     = 4.0f;
    s.FrameRounding     = 4.0f;
    s.PopupRounding     = 6.0f;
    s.ScrollbarRounding = 4.0f;
    s.GrabRounding      = 4.0f;
    s.TabRounding       = 4.0f;
    s.WindowPadding     = {8.0f, 8.0f};
    s.FramePadding      = {6.0f, 3.0f};
    s.ItemSpacing       = {6.0f, 5.0f};
    s.ItemInnerSpacing  = {4.0f, 4.0f};
    s.IndentSpacing     = 16.0f;
    s.ScrollbarSize     = 12.0f;
    s.GrabMinSize       = 8.0f;
    s.WindowBorderSize  = 1.0f;
    s.FrameBorderSize   = 0.0f;
    s.TabBorderSize     = 0.0f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_Text]                  = {0.92f, 0.92f, 0.96f, 1.00f};
    c[ImGuiCol_TextDisabled]          = {0.42f, 0.42f, 0.54f, 1.00f};
    c[ImGuiCol_WindowBg]              = {0.10f, 0.10f, 0.15f, 1.00f};
    c[ImGuiCol_ChildBg]               = {0.07f, 0.07f, 0.11f, 1.00f};
    c[ImGuiCol_PopupBg]               = {0.12f, 0.12f, 0.19f, 0.98f};
    c[ImGuiCol_Border]                = {0.20f, 0.20f, 0.30f, 1.00f};
    c[ImGuiCol_BorderShadow]          = {0.00f, 0.00f, 0.00f, 0.00f};
    c[ImGuiCol_FrameBg]               = {0.15f, 0.15f, 0.23f, 1.00f};
    c[ImGuiCol_FrameBgHovered]        = {0.21f, 0.21f, 0.31f, 1.00f};
    c[ImGuiCol_FrameBgActive]         = {0.26f, 0.44f, 0.70f, 1.00f};
    c[ImGuiCol_TitleBg]               = {0.07f, 0.07f, 0.11f, 1.00f};
    c[ImGuiCol_TitleBgActive]         = {0.11f, 0.21f, 0.38f, 1.00f};
    c[ImGuiCol_TitleBgCollapsed]      = {0.07f, 0.07f, 0.11f, 1.00f};
    c[ImGuiCol_MenuBarBg]             = {0.08f, 0.08f, 0.13f, 1.00f};
    c[ImGuiCol_ScrollbarBg]           = {0.07f, 0.07f, 0.11f, 1.00f};
    c[ImGuiCol_ScrollbarGrab]         = {0.20f, 0.20f, 0.31f, 1.00f};
    c[ImGuiCol_ScrollbarGrabHovered]  = {0.28f, 0.28f, 0.42f, 1.00f};
    c[ImGuiCol_ScrollbarGrabActive]   = {0.26f, 0.44f, 0.70f, 1.00f};
    c[ImGuiCol_CheckMark]             = {0.28f, 0.58f, 1.00f, 1.00f};
    c[ImGuiCol_SliderGrab]            = {0.28f, 0.58f, 1.00f, 1.00f};
    c[ImGuiCol_SliderGrabActive]      = {0.38f, 0.68f, 1.00f, 1.00f};
    c[ImGuiCol_Button]                = {0.16f, 0.28f, 0.50f, 1.00f};
    c[ImGuiCol_ButtonHovered]         = {0.24f, 0.42f, 0.68f, 1.00f};
    c[ImGuiCol_ButtonActive]          = {0.13f, 0.22f, 0.42f, 1.00f};
    c[ImGuiCol_Header]                = {0.16f, 0.28f, 0.50f, 0.80f};
    c[ImGuiCol_HeaderHovered]         = {0.24f, 0.42f, 0.68f, 0.80f};
    c[ImGuiCol_HeaderActive]          = {0.26f, 0.44f, 0.70f, 1.00f};
    c[ImGuiCol_Separator]             = {0.20f, 0.20f, 0.30f, 1.00f};
    c[ImGuiCol_SeparatorHovered]      = {0.26f, 0.44f, 0.70f, 0.80f};
    c[ImGuiCol_SeparatorActive]       = {0.26f, 0.44f, 0.70f, 1.00f};
    c[ImGuiCol_ResizeGrip]            = {0.16f, 0.28f, 0.50f, 0.40f};
    c[ImGuiCol_ResizeGripHovered]     = {0.24f, 0.42f, 0.68f, 0.80f};
    c[ImGuiCol_ResizeGripActive]      = {0.26f, 0.44f, 0.70f, 1.00f};
    c[ImGuiCol_Tab]                   = {0.11f, 0.15f, 0.25f, 1.00f};
    c[ImGuiCol_TabHovered]            = {0.24f, 0.42f, 0.68f, 1.00f};
    c[ImGuiCol_TabActive]             = {0.16f, 0.34f, 0.60f, 1.00f};
    c[ImGuiCol_TabUnfocused]          = {0.09f, 0.11f, 0.19f, 1.00f};
    c[ImGuiCol_TabUnfocusedActive]    = {0.13f, 0.24f, 0.44f, 1.00f};
    c[ImGuiCol_PlotLines]             = {0.28f, 0.58f, 1.00f, 1.00f};
    c[ImGuiCol_PlotLinesHovered]      = {0.38f, 0.68f, 1.00f, 1.00f};
    c[ImGuiCol_PlotHistogram]         = {0.28f, 0.58f, 1.00f, 1.00f};
    c[ImGuiCol_PlotHistogramHovered]  = {0.38f, 0.68f, 1.00f, 1.00f};
    c[ImGuiCol_TableHeaderBg]         = {0.11f, 0.21f, 0.38f, 1.00f};
    c[ImGuiCol_TableBorderStrong]     = {0.20f, 0.20f, 0.30f, 1.00f};
    c[ImGuiCol_TableBorderLight]      = {0.14f, 0.14f, 0.22f, 1.00f};
    c[ImGuiCol_NavHighlight]          = {0.26f, 0.44f, 0.70f, 1.00f};
    c[ImGuiCol_NavWindowingHighlight] = {0.26f, 0.44f, 0.70f, 0.70f};
}

// ─── Init ─────────────────────────────────────────────────────────────────────
void EditorUI::init(Scene* scene_, Workspace* workspace_, Camera* camera_,
                    std::vector<std::string>* models_, std::vector<std::string>* textures_,
                    std::vector<std::string>* audio_,
                    int* selectedModel_, int* selectedTexture_,
                    bool* useRayTracing_, bool* shadowsEnabled_, EditorCallbacks callbacks) {
    scene          = scene_;
    workspace      = workspace_;
    camera         = camera_;
    models         = models_;
    textures       = textures_;
    audio          = audio_;
    selectedModel  = selectedModel_;
    selectedTexture= selectedTexture_;
    useRayTracing  = useRayTracing_;
    shadowsEnabled = shadowsEnabled_;
    cb             = callbacks;
}

// ─── Main render ──────────────────────────────────────────────────────────────
void EditorUI::render(int& selectedEntityIndex, std::set<int>& selectedEntities,
                      bool& hasUnsavedChanges, std::string& currentSceneFile) {
    ImGuiIO& io = ImGui::GetIO();
    const Layout L = computeLayout(io.DisplaySize.x, io.DisplaySize.y);

    static ImGuizmo::OPERATION gizmoOp   = ImGuizmo::TRANSLATE;
    static ImGuizmo::MODE      gizmoMode = ImGuizmo::WORLD;
    const  ImGuizmo::OPERATION ROTATE_XYZ =
        (ImGuizmo::OPERATION)(ImGuizmo::ROTATE_X | ImGuizmo::ROTATE_Y | ImGuizmo::ROTATE_Z);
    if (gizmoOp == ImGuizmo::SCALE) gizmoMode = ImGuizmo::LOCAL;

    bool playing = cb.isPlaying();

    // Ortho scroll-to-zoom (when mouse is over the 3D viewport and not captured by ImGui)
    if (!playing && camera->orthoMode && !io.WantCaptureMouse) {
        float wheel = io.MouseWheel;
        if (wheel != 0.0f) {
            camera->orthoZoom = glm::clamp(camera->orthoZoom * (1.0f - wheel * 0.1f), 1.0f, 200.0f);
        }
    }

    // ── Gizmo (background draw list, constrained to viewport) ───────────────
    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
    ImGuizmo::SetRect(L.midX(), L.midY(), L.midW(), L.midH());

    if (selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size()) {
        GameObject& obj  = scene->entities[selectedEntityIndex];
        glm::mat4   view = camera->getViewMatrix();
        float       ar   = (L.midH() > 0.0f) ? (L.midW() / L.midH()) : 1.0f;
        glm::mat4   proj = camera->getProjMatrix(ar);
        glm::mat4   wm   = getWorldTransform(*scene, selectedEntityIndex);

        float snap[3] = {snapTranslate, snapTranslate, snapTranslate};
        if (gizmoOp == ROTATE_XYZ)         snap[0]=snap[1]=snap[2]=snapRotate;
        else if (gizmoOp==ImGuizmo::SCALE) snap[0]=snap[1]=snap[2]=snapScale;
        const float* snapPtr = snapEnabled ? snap : nullptr;

        ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                             gizmoOp, gizmoMode, glm::value_ptr(wm), nullptr, snapPtr);

        static bool wasUsing = false;
        bool isUsing = ImGuizmo::IsUsing();
        if (isUsing && !wasUsing) cb.requestUndoSnapshot();
        wasUsing = isUsing;

        if (isUsing) {
            glm::mat4 local = wm;
            if (obj.parentId) {
                for (int pi = 0; pi < (int)scene->entities.size(); pi++) {
                    if (scene->entities[pi].id == obj.parentId) {
                        local = glm::inverse(getWorldTransform(*scene, pi)) * wm;
                        break;
                    }
                }
            }
            glm::vec3 t, r, s;
            ImGuizmo::DecomposeMatrixToComponents(
                glm::value_ptr(local), glm::value_ptr(t), glm::value_ptr(r), glm::value_ptr(s));
            obj.transform.translation = t;
            obj.transform.rotation    = glm::radians(r);
            obj.transform.scale       = s;
            hasUnsavedChanges = true;
        }
    }

    // ── Toolbar (no title bar, always on top) ─────────────────────────────────
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({L.W, L.topH});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {6.0f, 6.0f});
    ImGui::Begin("##toolbar", nullptr,
        kDock | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();

    // Gizmo tool buttons  W / E / R
    auto toolBtn = [&](const char* lbl, ImGuizmo::OPERATION op) {
        bool active = (gizmoOp == op);
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, {0.26f, 0.44f, 0.70f, 1.0f});
        if (ImGui::Button(lbl, {28.0f, 22.0f})) gizmoOp = op;
        if (active) ImGui::PopStyleColor();
    };
    toolBtn("W##mv",  ImGuizmo::TRANSLATE);
    ImGui::SameLine(0, 2);
    toolBtn("E##rot", ROTATE_XYZ);
    ImGui::SameLine(0, 2);
    toolBtn("R##sc",  ImGuizmo::SCALE);

    // Local / World
    if (gizmoOp != ImGuizmo::SCALE) {
        ImGui::SameLine(0, 10);
        ImGui::TextDisabled("|");
        ImGui::SameLine(0, 10);
        auto modeBtn = [&](const char* lbl, ImGuizmo::MODE m) {
            bool a = (gizmoMode == m);
            if (a) ImGui::PushStyleColor(ImGuiCol_Button, {0.26f, 0.44f, 0.70f, 1.0f});
            if (ImGui::Button(lbl, {44.0f, 22.0f})) gizmoMode = m;
            if (a) ImGui::PopStyleColor();
        };
        modeBtn("Local", ImGuizmo::LOCAL);
        ImGui::SameLine(0, 2);
        modeBtn("World", ImGuizmo::WORLD);
    }

    // Snap
    ImGui::SameLine(0, 10);
    ImGui::TextDisabled("|");
    ImGui::SameLine(0, 10);
    ImGui::Checkbox("Snap", &snapEnabled);
    if (snapEnabled) {
        ImGui::SameLine(0, 6);
        ImGui::SetNextItemWidth(50); ImGui::DragFloat("##st", &snapTranslate, 0.05f, 0.01f, 10.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Translate snap");
        ImGui::SameLine(0, 3);
        ImGui::SetNextItemWidth(42); ImGui::DragFloat("##sr", &snapRotate, 1.0f, 1.0f, 90.0f, "%.0f\xc2\xb0");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotate snap (deg)");
        ImGui::SameLine(0, 3);
        ImGui::SetNextItemWidth(42); ImGui::DragFloat("##ss", &snapScale, 0.01f, 0.01f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scale snap");
    }

    // 2D / ortho toggle (editor-only)
    if (!playing) {
        ImGui::SameLine(0, 10);
        ImGui::TextDisabled("|");
        ImGui::SameLine(0, 10);
        bool ortho = camera->orthoMode;
        if (ortho) ImGui::PushStyleColor(ImGuiCol_Button, {0.26f, 0.44f, 0.70f, 1.0f});
        if (ImGui::Button("2D", {32.0f, 22.0f})) {
            camera->orthoMode = !camera->orthoMode;
            if (camera->orthoMode) {
                // Look straight down (-Z) with Y as screen-up
                camera->front = {0.0f, 0.0f, -1.0f};
                camera->up    = {0.0f, 1.0f,  0.0f};
            } else {
                // Restore a sensible 3D perspective direction
                camera->front = glm::normalize(glm::vec3{-1.0f, -1.0f, -1.0f});
                camera->up    = {0.0f, 0.0f, 1.0f};
            }
        }
        if (ortho) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle 2D/orthographic view");

        if (camera->orthoMode) {
            ImGui::SameLine(0, 4);
            ImGui::SetNextItemWidth(56);
            ImGui::DragFloat("##zoom", &camera->orthoZoom, 0.1f, 1.0f, 200.0f, "%.1f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ortho zoom (half-height)");
        }
    }

    // Play / Stop — always pinned to centre regardless of left-side width
    {
        const float btnW = 82.0f;
        const float cx   = (L.W - btnW) * 0.5f;
        ImGui::SameLine();
        ImGui::SetCursorPosX(cx);
        if (playing) {
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.60f, 0.10f, 0.10f, 1.0f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.80f, 0.15f, 0.15f, 1.0f});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  {0.44f, 0.07f, 0.07f, 1.0f});
            if (ImGui::Button("Stop", {btnW, 22.0f})) cb.togglePlayMode();
            ImGui::PopStyleColor(3);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.08f, 0.46f, 0.08f, 1.0f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.12f, 0.62f, 0.12f, 1.0f});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  {0.05f, 0.32f, 0.05f, 1.0f});
            if (ImGui::Button("Play", {btnW, 22.0f})) cb.togglePlayMode();
            ImGui::PopStyleColor(3);
        }
    }

    // Undo + Save + Export + scene name — always pinned to right edge
    {
        const float exportW = 58.0f;
        const float saveW   = 54.0f;
        const float undoW   = 54.0f;
        const float sepW    = 18.0f;
        const float nameW   = 140.0f;
        const float totalW  = sepW + nameW + undoW + 4.0f + saveW + 4.0f + exportW + 8.0f;
        ImGui::SameLine();
        ImGui::SetCursorPosX(L.W - totalW);
        ImGui::TextDisabled("|");
        ImGui::SameLine(0, 6);
        std::string title = currentSceneFile + (hasUnsavedChanges ? " *" : "");
        ImGui::SetNextItemWidth(nameW);
        ImGui::TextColored({0.65f, 0.80f, 1.0f, 1.0f}, "%.*s",
                           (int)std::min(title.size(), (size_t)28), title.c_str());
        ImGui::SameLine(0, 6);
        if (ImGui::Button("Undo", {undoW, 22.0f}) && cb.performUndo)
            cb.performUndo();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ctrl+Z");
        ImGui::SameLine(0, 4);
        if (ImGui::Button("Save", {saveW, 22.0f}) && !currentSceneFile.empty()) {
            cb.saveScene(currentSceneFile);
            hasUnsavedChanges = false;
        }
        ImGui::SameLine(0, 4);
        if (!playing && cb.exportProject) {
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.40f, 0.20f, 0.60f, 1.0f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.55f, 0.28f, 0.82f, 1.0f});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  {0.28f, 0.14f, 0.42f, 1.0f});
            if (ImGui::Button("Export", {exportW, 22.0f}))
                cb.exportProject(currentSceneFile);
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Export project to dist/<name>/ as a standalone bundle");
        }
    }
    ImGui::End();

    // Dim the editor panels during play mode
    if (playing) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.45f);

    // ── Left panel — Hierarchy + Project tabs ────────────────────────────────
    ImGui::SetNextWindowPos({0.0f, L.topH});
    ImGui::SetNextWindowSize({L.leftW, L.H - L.topH - L.botH});
    ImGui::Begin("##left", nullptr, kDock);

    if (ImGui::BeginTabBar("##ltabs")) {

        if (ImGui::BeginTabItem("Hierarchy")) {
            if (ImGui::Button("+ Entity",  {(L.leftW - 24) * 0.5f - 3, 0})) {
                cb.requestUndoSnapshot();
                GameObject e;
                e.id   = scene->nextEntityId++;
                e.name = "Entity_" + std::to_string(e.id);
                e.transform.translation = glm::vec3(0.0f, 0.0f, 0.0f);
                scene->entities.push_back(e);
                selectedEntityIndex = (int)scene->entities.size() - 1;
                hasUnsavedChanges = true;
            }
            ImGui::SameLine(0, 4);
            if (ImGui::Button("+ Terrain", {(L.leftW - 24) * 0.5f - 3, 0})) {
                cb.requestUndoSnapshot();
                GameObject t;
                t.id           = scene->nextEntityId++;
                t.name         = "Terrain_" + std::to_string(t.id);
                t.isTerrain    = true; t.terrainDirty = true;
                t.isStatic     = true; t.roughness    = 0.9f;
                scene->entities.push_back(t);
                selectedEntityIndex = (int)scene->entities.size() - 1;
                hasUnsavedChanges = true;
            }

            // Inline name edit for selected entity
            if (selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size()) {
                ImGui::Spacing();
                static char nameBuf[256] = {};
                strncpy(nameBuf, scene->entities[selectedEntityIndex].name.c_str(), sizeof(nameBuf)-1);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputText("##ename", nameBuf, sizeof(nameBuf),
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
                    cb.requestUndoSnapshot();
                    scene->entities[selectedEntityIndex].name = nameBuf;
                    hasUnsavedChanges = true;
                }
            }
            ImGui::Separator();

            static GameObject entityClipboard;
            static bool hasClipboard = false;

            // Keyboard shortcuts (entity tree must have focus or just always-on in Hierarchy tab)
            if (!playing) {
                if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_D) && selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size()) {
                    cb.requestUndoSnapshot();
                    GameObject copy = scene->entities[selectedEntityIndex];
                    copy.id = scene->nextEntityId++;
                    copy.name += "_copy";
                    copy.transform.translation += glm::vec3(0.5f, 0.0f, 0.0f);
                    scene->entities.push_back(copy);
                    selectedEntityIndex = (int)scene->entities.size() - 1;
                    hasUnsavedChanges = true;
                }
                if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C) && selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size()) {
                    entityClipboard = scene->entities[selectedEntityIndex];
                    hasClipboard = true;
                }
                if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V) && hasClipboard) {
                    cb.requestUndoSnapshot();
                    GameObject copy = entityClipboard;
                    copy.id = scene->nextEntityId++;
                    copy.name += "_copy";
                    copy.transform.translation += glm::vec3(0.5f, 0.0f, 0.0f);
                    scene->entities.push_back(copy);
                    selectedEntityIndex = (int)scene->entities.size() - 1;
                    hasUnsavedChanges = true;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Delete) && selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size() && !ImGui::IsAnyItemActive()) {
                    // handled below via pendingDel
                }
            }

            uint32_t pendingDel = 0;
            uint32_t pendingDup = 0;
            ImGui::BeginChild("##hier", {0, 0}, false);
            for (int i = 0; i < (int)scene->entities.size(); i++)
                if (scene->entities[i].parentId == 0)
                    renderEntityTree(scene, i, selectedEntityIndex, hasUnsavedChanges, pendingDel, pendingDup, pendingSavePrefabId);

            // Delete key inside hierarchy child window
            if (!playing && ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Delete) && selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size())
                pendingDel = scene->entities[selectedEntityIndex].id;

            ImGui::EndChild();

            if (pendingDup) {
                cb.requestUndoSnapshot();
                int si = -1;
                for (int i = 0; i < (int)scene->entities.size(); i++)
                    if (scene->entities[i].id == pendingDup) { si = i; break; }
                if (si >= 0) {
                    GameObject copy = scene->entities[si];
                    copy.id = scene->nextEntityId++;
                    copy.name += "_copy";
                    copy.transform.translation += glm::vec3(0.5f, 0.0f, 0.0f);
                    scene->entities.push_back(copy);
                    selectedEntityIndex = (int)scene->entities.size() - 1;
                    hasUnsavedChanges = true;
                }
            }

            if (pendingDel) {
                cb.requestUndoSnapshot();
                int di = -1;
                for (int i = 0; i < (int)scene->entities.size(); i++)
                    if (scene->entities[i].id == pendingDel) { di = i; break; }
                if (di >= 0) {
                    for (auto& e : scene->entities)
                        if (e.parentId == pendingDel) e.parentId = 0;
                    cb.onEntityDeleted(pendingDel);
                    scene->entities.erase(scene->entities.begin() + di);
                    selectedEntities.clear();
                    if (selectedEntityIndex == di)       selectedEntityIndex = -1;
                    else if (selectedEntityIndex > di)   selectedEntityIndex--;
                }
            }

            // Save-as-prefab modal — opened by context menu
            if (pendingSavePrefabId) {
                ImGui::OpenPopup("Save Prefab##modal");
                // Pre-fill with entity name if buffer is empty
                for (const auto& e : scene->entities) {
                    if (e.id == pendingSavePrefabId && prefabNameBuf[0] == '\0') {
                        strncpy(prefabNameBuf, e.name.c_str(), sizeof(prefabNameBuf)-1);
                        break;
                    }
                }
            }
            if (ImGui::BeginPopupModal("Save Prefab##modal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Prefab name:");
                ImGui::SetNextItemWidth(220);
                ImGui::InputText("##pname", prefabNameBuf, sizeof(prefabNameBuf));
                ImGui::Spacing();
                if (ImGui::Button("Save", {100, 0})) {
                    std::string name = prefabNameBuf;
                    if (name.empty()) name = "prefab";
                    if (name.find(".prefab") == std::string::npos) name += ".prefab";
                    std::string dir = workspace->activeProject.getPrefabsPath();
                    int rootIdx = -1;
                    for (int i = 0; i < (int)scene->entities.size(); i++)
                        if (scene->entities[i].id == pendingSavePrefabId) { rootIdx = i; break; }
                    if (rootIdx >= 0)
                        PrefabSerializer::save(*scene, rootIdx, dir + "/" + name);
                    pendingSavePrefabId = 0;
                    prefabNameBuf[0] = '\0';
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel", {100, 0})) {
                    pendingSavePrefabId = 0;
                    prefabNameBuf[0] = '\0';
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Project")) {
            ImGui::TextDisabled("Available Projects");
            if (ImGui::BeginListBox("##projs", {-1, 110})) {
                for (int i = 0; i < (int)workspace->availableProjects.size(); i++) {
                    bool sel = (selectedProjectIndex == i);
                    if (ImGui::Selectable(workspace->availableProjects[i].c_str(), sel))
                        selectedProjectIndex = i;
                    if (ImGui::BeginPopupContextItem()) {
                        ImGui::TextColored({1,0.3f,0.3f,1}, "Project Actions");
                        if (ImGui::Selectable("Delete Project")) {
                            std::string p = "projects/" + workspace->availableProjects[i];
                            if (std::filesystem::exists(p)) std::filesystem::remove_all(p);
                            cb.refreshProjects();
                            selectedProjectIndex = 0;
                        }
                        ImGui::EndPopup();
                    }
                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
                        cb.switchProject(workspace->availableProjects[i]);
                }
                ImGui::EndListBox();
            }
            if (ImGui::Button("Open Selected", {-1, 0}) && !workspace->availableProjects.empty())
                cb.switchProject(workspace->availableProjects[selectedProjectIndex]);
            ImGui::Separator();
            ImGui::SetNextItemWidth(-1);
            ImGui::InputTextWithHint("##np", "New project name...", newProjectNameBuffer,
                                     IM_ARRAYSIZE(newProjectNameBuffer));
            if (ImGui::Button("Create & Open", {-1, 0})) {
                cb.createProject(std::string(newProjectNameBuffer));
                newProjectNameBuffer[0] = '\0';
            }
            ImGui::Separator();
            if (ImGui::Button("Pack Assets (.rpak)", {-1, 0}) && cb.packAssets)
                cb.packAssets();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Pack all meshes and textures into a .rpak file for distribution.");
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    ImGui::End();

    // ── Right panel — Inspector ──────────────────────────────────────────────
    ImGui::SetNextWindowPos({L.W - L.rightW, L.topH});
    ImGui::SetNextWindowSize({L.rightW, L.H - L.topH - L.botH});
    ImGui::Begin("Inspector", nullptr, kDock);

    if (selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size()) {
        GameObject& obj = scene->entities[selectedEntityIndex];

        // Parent selector
        {
            std::string plabel = "(No Parent)";
            for (const auto& e : scene->entities)
                if (e.id == obj.parentId) { plabel = e.name; break; }
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##par", plabel.c_str())) {
                if (ImGui::Selectable("(No Parent)", obj.parentId==0)) {
                    obj.parentId = 0; hasUnsavedChanges = true;
                }
                for (int i = 0; i < (int)scene->entities.size(); i++) {
                    const auto& cand = scene->entities[i];
                    if (cand.id == obj.id) continue;
                    // Cycle guard
                    bool cycle = false;
                    uint32_t chk = cand.parentId;
                    for (int d = 0; d < 64 && chk; d++) {
                        if (chk == obj.id) { cycle = true; break; }
                        bool found = false;
                        for (const auto& e2 : scene->entities)
                            if (e2.id == chk) { chk = e2.parentId; found = true; break; }
                        if (!found) break;
                    }
                    if (cycle) continue;
                    if (ImGui::Selectable(cand.name.c_str(), obj.parentId==cand.id)) {
                        obj.parentId = cand.id; hasUnsavedChanges = true;
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::TextDisabled("Parent");
        }
        ImGui::Spacing();

        InspCtx ctx{scene, models, textures, audio, selectedModel, selectedTexture,
                    cb, hasUnsavedChanges};

        inspTransform(obj, cb, hasUnsavedChanges);
        inspMeshMaterial(obj, ctx);
        inspPhysics(obj, hasUnsavedChanges);
        inspLight(obj, scene, hasUnsavedChanges);
        inspTerrain(obj, hasUnsavedChanges);
        inspAudio(obj, audio, hasUnsavedChanges);
        inspParticles(obj, hasUnsavedChanges);
        inspAnimation(obj, models, cb, hasUnsavedChanges);
        inspAI(obj, scene, hasUnsavedChanges);
        inspUICanvas(obj, hasUnsavedChanges);
        inspScripting(obj, cb, hasUnsavedChanges,
                      showScriptEditor, scriptEditorContent, scriptEditorFile, scriptEditorDirty,
                      scriptEditorEntityId, selectedEntityIndex,
                      showVisualEditor, visualEditorEntityId);
    } else {
        ImGui::Spacing();
        ImGui::TextDisabled("  Select an entity in the Hierarchy.");
    }
    ImGui::End();

    // ── Bottom panel — Console / Assets / Scene / Environment tabs ──────────
    ImGui::SetNextWindowPos({0.0f, L.H - L.botH});
    ImGui::SetNextWindowSize({L.W, L.botH});
    ImGui::Begin("##bottom", nullptr, kDock);

    if (ImGui::BeginTabBar("##btabs")) {

        // ── Console ──────────────────────────────────────────────────────────
        if (ImGui::BeginTabItem("Console")) {
            static bool   autoScroll   = true;
            static bool   showInfo     = true;
            static bool   showWarning  = true;
            static bool   showError    = true;
            static char   logFilter[128] = {};

            if (ImGui::Button("Clear")) EngineLog::get().clear();
            ImGui::SameLine();
            if (ImGui::Button("Copy All")) {
                std::string all;
                for (const auto& e : EngineLog::get().entries) {
                    const char* lv = e.level==LogLevel::Error  ?"ERROR":
                                     e.level==LogLevel::Warning?"WARN ":"INFO ";
                    all += "["+e.timestamp+"]["+lv+"] "+e.file+":"+
                           std::to_string(e.line)+"  "+e.message+"\n";
                }
                ImGui::SetClipboardText(all.c_str());
            }
            ImGui::SameLine();
            ImGui::Checkbox("Auto-scroll", &autoScroll);
            ImGui::SameLine(0, 12);
            ImGui::SetNextItemWidth(140);
            ImGui::InputTextWithHint("##lf", "Filter...", logFilter, sizeof(logFilter));
            ImGui::SameLine(0, 6);
            ImGui::Checkbox("Info",  &showInfo);
            ImGui::SameLine(0, 4);
            ImGui::Checkbox("Warn",  &showWarning);
            ImGui::SameLine(0, 4);
            ImGui::Checkbox("Error", &showError);
            ImGui::Separator();

            ImGui::BeginChild("##logscroll", {0, 0}, false, ImGuiWindowFlags_HorizontalScrollbar);
            for (const auto& entry : EngineLog::get().entries) {
                if (entry.level==LogLevel::Info    && !showInfo)    continue;
                if (entry.level==LogLevel::Warning && !showWarning) continue;
                if (entry.level==LogLevel::Error   && !showError)   continue;
                if (logFilter[0] && entry.message.find(logFilter)==std::string::npos
                                 && entry.file.find(logFilter)==std::string::npos) continue;
                ImVec4 col;
                switch (entry.level) {
                    case LogLevel::Warning: col={1.0f,0.82f,0.0f,1.0f}; break;
                    case LogLevel::Error:   col={1.0f,0.32f,0.32f,1.0f}; break;
                    default:                col={0.72f,0.72f,0.76f,1.0f}; break;
                }
                ImGui::TextDisabled("[%s] %s:%d", entry.timestamp.c_str(),
                                    entry.file.c_str(), entry.line);
                ImGui::SameLine();
                ImGui::TextColored(col, "%s", entry.message.c_str());
            }
            if (autoScroll && EngineLog::get().scrollToBottom) {
                ImGui::SetScrollHereY(1.0f);
                EngineLog::get().scrollToBottom = false;
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ── Assets ───────────────────────────────────────────────────────────
        if (ImGui::BeginTabItem("Assets")) {
            if (ImGui::Button("Refresh")) cb.refreshAssets();
            ImGui::SameLine();
            ImGui::TextDisabled("Scans project assets/ folders");
            ImGui::Separator();

            float half = (L.W - 20) * 0.5f;

            ImGui::BeginChild("##assetL", {half, 0}, true);
            ImGui::TextColored({0.65f,0.80f,1.0f,1.0f}, "Models");
            if (models && !models->empty()) {
                std::vector<const char*> ns;
                for (const auto& m : *models) ns.push_back(m.c_str());
                ImGui::SetNextItemWidth(-1);
                ImGui::ListBox("##mlist", selectedModel, ns.data(), (int)ns.size(), 5);
                if (ImGui::Button("Apply Model", {-1, 0})) { cb.applyModel(); hasUnsavedChanges = true; }
            } else {
                ImGui::TextDisabled("No models found");
            }
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("##assetR", {0, 0}, true);
            ImGui::TextColored({0.65f,0.80f,1.0f,1.0f}, "Textures");
            if (textures && !textures->empty()) {
                std::vector<const char*> ns;
                for (const auto& t : *textures) ns.push_back(t.c_str());
                ImGui::SetNextItemWidth(-1);
                ImGui::ListBox("##tlist", selectedTexture, ns.data(), (int)ns.size(), 5);
                if (ImGui::Button("Apply Texture", {-1, 0})) {
                    if (selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size())
                        scene->entities[selectedEntityIndex].textureIndex = *selectedTexture;
                    hasUnsavedChanges = true;
                }
            } else {
                ImGui::TextDisabled("No textures found");
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ── Scene ─────────────────────────────────────────────────────────────
        if (ImGui::BeginTabItem("Scene")) {
            static char sceneNameBuf[128] = "";
            if (sceneNameBuf[0] == '\0')
                strncpy(sceneNameBuf, currentSceneFile.c_str(), sizeof(sceneNameBuf)-1);

            ImGui::TextColored({0.65f,0.80f,1.0f,1.0f}, "%s%s",
                               currentSceneFile.c_str(), hasUnsavedChanges ? " *" : "");

            // Collect scene files
            std::vector<std::string> scenes;
            std::string sPath = workspace->activeProject.getScenesPath();
            if (std::filesystem::exists(sPath))
                for (const auto& entry : std::filesystem::directory_iterator(sPath))
                    if (entry.path().extension() == ".scene")
                        scenes.push_back(entry.path().filename().string());

            float half = (L.W - 20) * 0.5f;
            ImGui::BeginChild("##scL", {half, 0}, true);
            ImGui::TextDisabled("File name");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##sname", sceneNameBuf, sizeof(sceneNameBuf));
            ImGui::Spacing();
            if (ImGui::Button("Save Scene", {-1, 0})) {
                std::string tgt = sceneNameBuf;
                if (tgt.find(".scene") == std::string::npos) tgt += ".scene";
                bool exists = false;
                for (const auto& s : scenes) if (s == tgt) exists = true;
                if (exists && tgt != currentSceneFile) {
                    showOverwriteModal = true; pendingSceneToLoad = tgt;
                } else {
                    cb.saveScene(tgt); currentSceneFile = tgt; hasUnsavedChanges = false;
                    strncpy(sceneNameBuf, tgt.c_str(), sizeof(sceneNameBuf)-1);
                }
            }
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::BeginChild("##scR", {0, 0}, true);
            ImGui::TextDisabled("Saved scenes");
            for (const auto& s : scenes) {
                bool sel = (s == std::string(sceneNameBuf));
                if (ImGui::Selectable(s.c_str(), sel))
                    strncpy(sceneNameBuf, s.c_str(), sizeof(sceneNameBuf)-1);
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::Selectable("Delete")) {
                        std::string fp = sPath + "/" + s;
                        if (std::filesystem::exists(fp)) std::filesystem::remove(fp);
                        if (currentSceneFile == s) {
                            currentSceneFile = "untitled.scene"; hasUnsavedChanges = true;
                        }
                        sceneNameBuf[0] = '\0';
                    }
                    ImGui::EndPopup();
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    if (hasUnsavedChanges) { showUnsavedModal = true; pendingSceneToLoad = s; }
                    else { cb.loadScene(s); currentSceneFile = s; hasUnsavedChanges = false; }
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ── Environment ───────────────────────────────────────────────────────
        if (ImGui::BeginTabItem("Environment")) {
            float half = (L.W - 20) * 0.5f;
            ImGui::BeginChild("##envL", {half, 0}, false);
            if (ImGui::ColorEdit3("Sky Color",     &scene->skyColor.x))     hasUnsavedChanges = true;
            if (ImGui::ColorEdit3("Ambient Light", &scene->ambientLight.x)) hasUnsavedChanges = true;
            if (ImGui::DragFloat3("Sun Direction",  &scene->sunDirection.x, 0.05f, -2.0f, 2.0f))
                hasUnsavedChanges = true;
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::BeginChild("##envR", {0, 0}, false);
            if (ImGui::Checkbox("Shadows",     shadowsEnabled))   hasUnsavedChanges = true;
            if (ImGui::Checkbox("Ray Tracing", useRayTracing)) {}
            if (*useRayTracing && !cb.isRayTracingReady()) {
                ImGui::TextColored({1.0f,0.2f,0.2f,1.0f}, "RTX unavailable!");
                *useRayTracing = false;
            }
            if (*useRayTracing)
                ImGui::TextDisabled("Shadows disabled in RT mode.");
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ── Prefabs ───────────────────────────────────────────────────────────
        if (ImGui::BeginTabItem("Prefabs")) {
            std::string prefabDir = workspace->activeProject.getPrefabsPath();
            std::vector<std::string> prefabs;
            if (std::filesystem::exists(prefabDir))
                for (const auto& entry : std::filesystem::directory_iterator(prefabDir))
                    if (entry.path().extension() == ".prefab")
                        prefabs.push_back(entry.path().filename().string());

            if (prefabs.empty()) {
                ImGui::TextDisabled("No prefabs yet.");
                ImGui::TextDisabled("Right-click an entity in the Hierarchy to save one.");
            } else {
                ImGui::TextDisabled("%d prefab(s) in project", (int)prefabs.size());
                ImGui::Separator();
                float half = (L.W - 20) * 0.5f;
                ImGui::BeginChild("##prefabList", {half, 0}, true);
                for (int i = 0; i < (int)prefabs.size(); i++) {
                    bool sel = (selectedPrefabIdx == i);
                    if (ImGui::Selectable(prefabs[i].c_str(), sel))
                        selectedPrefabIdx = i;
                    if (ImGui::BeginPopupContextItem()) {
                        if (ImGui::MenuItem("Delete")) {
                            std::string fp = prefabDir + "/" + prefabs[i];
                            if (std::filesystem::exists(fp)) std::filesystem::remove(fp);
                            selectedPrefabIdx = -1;
                        }
                        ImGui::EndPopup();
                    }
                }
                ImGui::EndChild();
                ImGui::SameLine();
                ImGui::BeginChild("##prefabActions", {0, 0}, true);
                if (selectedPrefabIdx >= 0 && selectedPrefabIdx < (int)prefabs.size()) {
                    ImGui::TextColored({0.65f,0.90f,0.65f,1.0f}, "%s", prefabs[selectedPrefabIdx].c_str());
                    ImGui::Spacing();
                    if (ImGui::Button("Instantiate", {-1, 0})) {
                        std::string path = prefabDir + "/" + prefabs[selectedPrefabIdx];
                        cb.requestUndoSnapshot();
                        int idx = PrefabSerializer::instantiate(*scene, path);
                        if (idx >= 0) {
                            selectedEntityIndex = idx;
                            selectedEntities.clear();
                            selectedEntities.insert(idx);
                            hasUnsavedChanges = true;
                        }
                    }
                } else {
                    ImGui::TextDisabled("Select a prefab to instantiate.");
                }
                ImGui::EndChild();
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    ImGui::End();

    if (playing) ImGui::PopStyleVar();  // Alpha

    // ── Script editor (floating) ─────────────────────────────────────────────
    if (showScriptEditor) {
        ImGui::SetNextWindowSize({680, 480}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos({L.midX() + 20, L.midY() + 20}, ImGuiCond_FirstUseEver);
        std::string edTitle = "Script — " + scriptEditorFile +
                              (scriptEditorDirty ? " *" : "") + "##se";
        if (ImGui::Begin(edTitle.c_str(), &showScriptEditor)) {
            // Sync staging buffer when file changes
            static char   stageBuf[65536] = {};
            static std::string stageFor;
            if (stageFor != scriptEditorFile) {
                strncpy(stageBuf, scriptEditorContent.c_str(), sizeof(stageBuf)-1);
                stageBuf[sizeof(stageBuf)-1] = '\0';
                stageFor = scriptEditorFile;
            }
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float  btnH  = ImGui::GetFrameHeightWithSpacing() + 4.0f;
            if (ImGui::InputTextMultiline("##code", stageBuf, sizeof(stageBuf),
                                          {avail.x, avail.y - btnH},
                                          ImGuiInputTextFlags_AllowTabInput)) {
                scriptEditorContent = stageBuf;
                scriptEditorDirty   = true;
            }
            if (ImGui::Button("Save") && cb.writeScriptFile) {
                cb.writeScriptFile(scriptEditorFile, scriptEditorContent);
                scriptEditorDirty = false;
                if (cb.isPlaying && cb.isPlaying() && cb.hotReloadScript && scriptEditorEntityId != 0)
                    cb.hotReloadScript(scriptEditorEntityId, scriptEditorContent);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Ctrl+Tab inserts a tab character");
        }
        ImGui::End();
    }

    // ── Visual editor floating window ─────────────────────────────────────────
    renderVisualEditorWindow();

    // ── Modals ────────────────────────────────────────────────────────────────
    if (showOverwriteModal) ImGui::OpenPopup("Overwrite File?");
    if (ImGui::BeginPopupModal("Overwrite File?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("'%s' already exists. Overwrite?", pendingSceneToLoad.c_str());
        ImGui::Separator();
        if (ImGui::Button("Overwrite", {120, 0})) {
            cb.saveScene(pendingSceneToLoad);
            currentSceneFile = pendingSceneToLoad;
            hasUnsavedChanges = false;
            showOverwriteModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", {120, 0})) {
            showOverwriteModal = false; ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (showUnsavedModal) ImGui::OpenPopup("Unsaved Changes");
    if (ImGui::BeginPopupModal("Unsaved Changes", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("You have unsaved changes.\nLoad '%s' anyway?", pendingSceneToLoad.c_str());
        ImGui::Separator();
        if (ImGui::Button("Discard & Load", {140, 0})) {
            cb.loadScene(pendingSceneToLoad);
            currentSceneFile = pendingSceneToLoad;
            hasUnsavedChanges = false;
            showUnsavedModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", {120, 0})) {
            showUnsavedModal = false; ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// ─── Visual script editor (Scratch-like floating window) ──────────────────────
void EditorUI::renderVisualEditorWindow() {
    if (!showVisualEditor) return;

    // Find entity
    GameObject* obj = nullptr;
    for (auto& e : scene->entities)
        if (e.id == visualEditorEntityId) { obj = &e; break; }
    if (!obj) { showVisualEditor = false; return; }

    VisualScript& vs = obj->visualScript;

    // Block category colors
    static const ImU32 kCEvt  = IM_COL32(250,171,25,255);   // amber  – events
    static const ImU32 kCMov  = IM_COL32(71,143,242,255);   // blue   – motion
    static const ImU32 kCSnds = IM_COL32(158,97,242,255);   // purple – sound
    static const ImU32 kCLok  = IM_COL32(78,185,185,255);   // teal   – looks
    static const ImU32 kCTxt  = IM_COL32(71,191,107,255);   // green  – text
    static const ImU32 kCCtl  = IM_COL32(242,122,20,255);   // orange – control
    static const ImU32 kCUI   = IM_COL32(107,158,224,255);  // l-blue – ui

    // Per-action color and param metadata
    static const ImU32 kActCol[] = {
        kCLok,  // Log
        kCSnds, // PlaySound
        kCMov,  // SetPosition
        kCMov,  // MoveToward
        kCLok,  // SetVisible
        kCMov,  // SetRotation
        kCCtl,  // LoadScene
        kCTxt,  // ShowDialogue
        kCCtl,  // Wait
        kCUI,   // SetUIText
        kCUI,   // SetUIValue
    };
    static const char* kActNames[] = {
        "Log","Play Sound","Set Position","Move Toward",
        "Set Visible","Set Rotation","Load Scene",
        "Show Dialogue","Wait","Set UI Text","Set UI Value"
    };
    static const char* kEvtNames[] = {
        "On Start","On Update","On Interact","On Trigger Enter","On Trigger Exit"
    };
    static const char* kParamLabel[][5] = {
        {"Message","","","",""},
        {"File","","","",""},
        {"Entity","X","Y","Z",""},
        {"Entity","X","Y","Z","Speed"},
        {"Entity","Visible","","",""},
        {"Entity","RX","RY","RZ",""},
        {"Scene","","","",""},
        {"Speaker","Text","","",""},
        {"Seconds","","","",""},
        {"Entity","Elem#","Text","",""},
        {"Entity","Elem#","Value","",""},
    };
    static const int kParamCnt[] = {1,1,4,5,2,4,1,2,1,3,3};

    // Geometry constants
    static constexpr float kPalW    = 165.0f;
    static constexpr float kBlkW    = 340.0f;
    static constexpr float kHatH    = 34.0f;  // event hat block height
    static constexpr float kHdrH    = 28.0f;  // action header row height
    static constexpr float kPrmH    = 22.0f;  // per-param row height
    static constexpr float kPadBot  = 5.0f;
    static constexpr float kNotchW  = 24.0f;
    static constexpr float kNotchX  = 16.0f;
    static constexpr float kNotchH  = 7.0f;
    static constexpr float kBlkR    = 6.0f;

    auto darken = [](ImU32 c) -> ImU32 {
        return IM_COL32((int)((c      &0xff)*0.60f),
                        (int)(((c>>8) &0xff)*0.60f),
                        (int)(((c>>16)&0xff)*0.60f), 255);
    };

    ImGui::SetNextWindowSize({kPalW + kBlkW + 50.0f, 580.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos({120.0f, 80.0f}, ImGuiCond_FirstUseEver);
    char title[128];
    snprintf(title, sizeof(title), "Visual Script \xe2\x80\x94 %s##vsw", obj->name.c_str());
    bool open = true;
    if (!ImGui::Begin(title, &open)) { ImGui::End(); if (!open) showVisualEditor = false; return; }
    if (!open) { showVisualEditor = false; ImGui::End(); return; }

    bool changed = false;

    // ── Palette (left) ────────────────────────────────────────────────────────
    ImGui::BeginChild("##vspal", {kPalW, 0}, true);
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float iw = kPalW - 16.0f;

        // Helper: draw a palette block button
        auto palBtn = [&](const char* label, ImU32 col) -> bool {
            ImVec2 p  = ImGui::GetCursorScreenPos();
            float  bh = 22.0f;
            dl->AddRectFilled(p, {p.x+iw, p.y+bh}, col, 5.0f);
            dl->AddRect      (p, {p.x+iw, p.y+bh}, IM_COL32(0,0,0,70), 5.0f, 0, 1.2f);
            ImVec2 ts = ImGui::CalcTextSize(label);
            dl->AddText({p.x+(iw-ts.x)*0.5f, p.y+(bh-ts.y)*0.5f},
                        IM_COL32(255,255,255,240), label);
            ImGui::InvisibleButton(label, {iw, bh});
            bool clicked = ImGui::IsItemClicked();
            ImGui::Spacing();
            return clicked;
        };

        auto catHdr = [](const char* label, ImVec4 col) {
            ImGui::TextColored(col, "%s", label);
        };

        catHdr("Events", {0.98f,0.67f,0.10f,1.0f});
        for (int i = 0; i < 5; i++) {
            ImGui::PushID(i);
            if (palBtn(kEvtNames[i], kCEvt)) {
                VSBlock blk; blk.event = (VSEventType)i;
                vs.blocks.push_back(blk);
                vsActiveStack = (int)vs.blocks.size()-1;
                changed = true;
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
        catHdr("Motion", {0.44f,0.72f,0.98f,1.0f});
        int motionActs[] = {2, 3, 5};
        for (int ai : motionActs) {
            ImGui::PushID(100+ai);
            if (palBtn(kActNames[ai], kActCol[ai]) && !vs.blocks.empty()) {
                int si = std::clamp(vsActiveStack,0,(int)vs.blocks.size()-1);
                VSAction a; a.type=(VSActionType)ai; vs.blocks[si].actions.push_back(a);
                changed=true;
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
        catHdr("Sound", {0.72f,0.50f,0.98f,1.0f});
        ImGui::PushID(101);
        if (palBtn(kActNames[1], kActCol[1]) && !vs.blocks.empty()) {
            int si=std::clamp(vsActiveStack,0,(int)vs.blocks.size()-1);
            VSAction a; a.type=VSActionType::PlaySound; vs.blocks[si].actions.push_back(a);
            changed=true;
        }
        ImGui::PopID();
        ImGui::Spacing();
        catHdr("Looks", {0.38f,0.85f,0.85f,1.0f});
        int looksActs[] = {0, 4};
        for (int ai : looksActs) {
            ImGui::PushID(200+ai);
            if (palBtn(kActNames[ai], kActCol[ai]) && !vs.blocks.empty()) {
                int si=std::clamp(vsActiveStack,0,(int)vs.blocks.size()-1);
                VSAction a; a.type=(VSActionType)ai; vs.blocks[si].actions.push_back(a);
                changed=true;
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
        catHdr("Text", {0.42f,0.88f,0.55f,1.0f});
        ImGui::PushID(107);
        if (palBtn(kActNames[7], kActCol[7]) && !vs.blocks.empty()) {
            int si=std::clamp(vsActiveStack,0,(int)vs.blocks.size()-1);
            VSAction a; a.type=VSActionType::ShowDialogue; vs.blocks[si].actions.push_back(a);
            changed=true;
        }
        ImGui::PopID();
        ImGui::Spacing();
        catHdr("Control", {0.95f,0.60f,0.28f,1.0f});
        int ctlActs[] = {6, 8};
        for (int ai : ctlActs) {
            ImGui::PushID(300+ai);
            if (palBtn(kActNames[ai], kActCol[ai]) && !vs.blocks.empty()) {
                int si=std::clamp(vsActiveStack,0,(int)vs.blocks.size()-1);
                VSAction a; a.type=(VSActionType)ai; vs.blocks[si].actions.push_back(a);
                changed=true;
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
        catHdr("UI", {0.55f,0.75f,0.95f,1.0f});
        for (int ai = 9; ai <= 10; ai++) {
            ImGui::PushID(400+ai);
            if (palBtn(kActNames[ai], kActCol[ai]) && !vs.blocks.empty()) {
                int si=std::clamp(vsActiveStack,0,(int)vs.blocks.size()-1);
                VSAction a; a.type=(VSActionType)ai; vs.blocks[si].actions.push_back(a);
                changed=true;
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();

    // ── Canvas (right) ────────────────────────────────────────────────────────
    ImGui::BeginChild("##vscanvas", {0,0}, true, ImGuiWindowFlags_HorizontalScrollbar);
    {
        if (vs.blocks.empty())
            ImGui::TextDisabled("Click an event in the palette to start.");

        int blockToRemove = -1;
        for (int bi = 0; bi < (int)vs.blocks.size(); bi++) {
            VSBlock& blk = vs.blocks[bi];
            ImGui::PushID(bi);
            bool isActive = (bi == vsActiveStack);

            // ── Hat block (event) ────────────────────────────────────────────
            {
                ImVec2 bp = ImGui::GetCursorScreenPos();
                float  bw = kBlkW;
                ImDrawList* dl = ImGui::GetWindowDrawList();

                // Active glow
                if (isActive)
                    dl->AddRect({bp.x-2,bp.y-2},{bp.x+bw+2,bp.y+kHatH+2},
                                IM_COL32(255,255,255,110), kBlkR+2, 0, 2.0f);

                // Hat body (fully rounded)
                dl->AddRectFilled(bp,{bp.x+bw,bp.y+kHatH}, kCEvt, kBlkR);
                // Output notch at bottom
                dl->AddRectFilled({bp.x+kNotchX, bp.y+kHatH},
                                  {bp.x+kNotchX+kNotchW, bp.y+kHatH+kNotchH},
                                  kCEvt, 0.0f);

                // Widgets inside hat
                float wy = bp.y + (kHatH - ImGui::GetFrameHeight()) * 0.5f;
                ImGui::SetCursorScreenPos({bp.x+8.0f, wy});
                ImGui::PushStyleColor(ImGuiCol_FrameBg,   IM_COL32(0,0,0,80));
                ImGui::PushStyleColor(ImGuiCol_Text,      IM_COL32(255,255,255,255));
                ImGui::SetNextItemWidth(148.0f);
                int ev = (int)blk.event;
                if (ImGui::Combo("##evt", &ev, kEvtNames, 5)) { blk.event=(VSEventType)ev; changed=true; }
                ImGui::PopStyleColor(2);
                ImGui::SameLine(0,6);
                if (isActive) {
                    ImGui::PushStyleColor(ImGuiCol_Button,{0.26f,0.44f,0.70f,1.0f});
                    ImGui::SmallButton("Active");
                    ImGui::PopStyleColor();
                } else {
                    if (ImGui::SmallButton("Set Active")) vsActiveStack = bi;
                }
                ImGui::SameLine(0,4);
                if (ImGui::SmallButton("x##hb")) { blockToRemove=bi; changed=true; }

                // Advance cursor past hat + notch
                ImGui::SetCursorScreenPos({bp.x, bp.y+kHatH+kNotchH});
                ImGui::Dummy({bw, 0.0f});
            }

            // ── Action blocks ────────────────────────────────────────────────
            int actToRemove = -1;
            for (int ai = 0; ai < (int)blk.actions.size(); ai++) {
                VSAction& act = blk.actions[ai];
                ImGui::PushID(ai);
                int at = (int)act.type;
                if (at < 0 || at > 10) at = 0;
                int  np    = kParamCnt[at];
                ImU32 col  = kActCol[at];
                ImU32 dark = darken(col);
                float bh   = kHdrH + np*kPrmH + kPadBot;
                bool  last = (ai == (int)blk.actions.size()-1);

                ImVec2    bp = ImGui::GetCursorScreenPos();
                float     bw = kBlkW;
                ImDrawList* dl = ImGui::GetWindowDrawList();

                // Body: square top, rounded bottom
                dl->AddRectFilled(bp, {bp.x+bw, bp.y+bh},
                                  col, kBlkR, ImDrawFlags_RoundCornersBottom);
                dl->AddRectFilled(bp, {bp.x+bw, bp.y+kBlkR}, col, 0.0f);

                // Slot indent at top
                dl->AddRectFilled({bp.x+kNotchX, bp.y},
                                  {bp.x+kNotchX+kNotchW, bp.y+kNotchH}, dark, 0.0f);

                // Output notch (except last in stack)
                if (!last)
                    dl->AddRectFilled({bp.x+kNotchX, bp.y+bh},
                                      {bp.x+kNotchX+kNotchW, bp.y+bh+kNotchH}, col, 0.0f);

                // Header row: combo + reorder + delete
                float wy = bp.y + (kHdrH-ImGui::GetFrameHeight())*0.5f;
                ImGui::SetCursorScreenPos({bp.x+8.0f, wy});
                ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0,0,0,80));
                ImGui::PushStyleColor(ImGuiCol_Text,    IM_COL32(255,255,255,255));
                ImGui::SetNextItemWidth(148.0f);
                if (ImGui::Combo("##at",&at, kActNames, 11)) { act.type=(VSActionType)at; changed=true; }
                ImGui::PopStyleColor(2);
                ImGui::SameLine(0,4);
                if (ImGui::SmallButton("^") && ai > 0) {
                    std::swap(blk.actions[ai-1], blk.actions[ai]); changed=true;
                }
                ImGui::SameLine(0,2);
                if (ImGui::SmallButton("v") && !last) {
                    std::swap(blk.actions[ai], blk.actions[ai+1]); changed=true;
                }
                ImGui::SameLine(0,2);
                if (ImGui::SmallButton("x##act")) { actToRemove=ai; changed=true; }

                // Param rows
                int at2 = (int)act.type;
                if (at2 < 0 || at2 > 10) at2 = 0;
                int np2 = kParamCnt[at2];
                for (int pi = 0; pi < np2; pi++) {
                    float py = bp.y + kHdrH + pi*kPrmH;
                    ImGui::SetCursorScreenPos({bp.x+8.0f, py});
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255,255,255,200));
                    ImGui::TextUnformatted(kParamLabel[at2][pi]);
                    ImGui::PopStyleColor();
                    ImGui::SameLine(0,0);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, 64.0f - ImGui::CalcTextSize(kParamLabel[at2][pi]).x));
                    char pid[64]; snprintf(pid,sizeof(pid),"##p%d_%d_%d",bi,ai,pi);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
                    ImGui::SetNextItemWidth(bw - 82.0f);
                    char pbuf[256]={};
                    strncpy(pbuf, act.p[pi].c_str(), sizeof(pbuf)-1);
                    if (ImGui::InputText(pid,pbuf,sizeof(pbuf))) { act.p[pi]=pbuf; changed=true; }
                    ImGui::PopStyleVar();
                }

                // Advance cursor past block + notch
                float advance = bh + (last ? 0.0f : kNotchH);
                ImGui::SetCursorScreenPos({bp.x, bp.y + advance});
                ImGui::Dummy({bw, 0.0f});

                ImGui::PopID();
            }
            if (actToRemove >= 0) blk.actions.erase(blk.actions.begin()+actToRemove);

            // "+ Action" button at bottom of stack
            if (ImGui::Button("+ Action##ba", {100.0f,0})) {
                blk.actions.push_back({}); vsActiveStack=bi; changed=true;
            }
            ImGui::Spacing(); ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PopID();
        }
        if (blockToRemove >= 0) {
            vs.blocks.erase(vs.blocks.begin()+blockToRemove);
            if (vsActiveStack >= (int)vs.blocks.size())
                vsActiveStack = std::max(0, (int)vs.blocks.size()-1);
            changed = true;
        }
    }
    ImGui::EndChild();
    ImGui::End();

    if (changed) {
        if (obj->luaScriptFile.empty()) obj->luaScriptFile = obj->name + ".lua";
        std::string code = compileVisualScript(vs);
        if (cb.writeScriptFile) cb.writeScriptFile(obj->luaScriptFile, code);
        if (cb.isPlaying && cb.isPlaying() && cb.hotReloadScript)
            cb.hotReloadScript(obj->id, code);
    }
}
