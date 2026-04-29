#pragma once
#include "Scene.hpp"
#include <string>

namespace SceneSerializer {
    // fullPath must include the file extension (e.g. "projects/Foo/scenes/level.scene")
    void save(const Scene& scene, const std::string& fullPath);
    Scene load(const std::string& fullPath);
}
