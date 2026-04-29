cmake_minimum_required(VERSION 3.15)
project(RPGMaker3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23) #17

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# This tells the compiler it can find <imgui.h> inside src/imgui
include_directories(src/imgui)
include_directories(src/glm)
add_executable(RPGMaker3D
    src/main.cpp
    src/ProjectManager.cpp
    src/SceneSerializer.cpp
    src/EditorUI.cpp
    src/VulkanEngine.cpp
    src/VulkanSetup.cpp
    src/VulkanInput.cpp
    src/VulkanAssets.cpp
    src/imgui/imgui.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_impl_glfw.cpp
    src/imgui/imgui_impl_vulkan.cpp
    src/ImGuizmo.cpp
    src/RayTracer.cpp
    src/AudioEngine.cpp
    src/LuaEngine.cpp
    src/Runtime.cpp
    src/Editor.cpp
    src/RendererDraw.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw dl pthread m lua5.4)
