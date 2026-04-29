#!/bin/bash
set -e

echo "=== [1/3] RECOMPILARE SHADERE ==="
glslc shaders/shader.vert -o shaders/vert.spv
glslc shaders/shader.frag -o shaders/frag.spv
glslc shaders/post.vert -o shaders/post.vert.spv
glslc shaders/post.frag -o shaders/post.frag.spv
glslc shaders/particle.vert -o shaders/particle.vert.spv
glslc shaders/particle.frag -o shaders/particle.frag.spv
glslc shaders/skinned.vert -o shaders/skinned.vert.spv
glslc shaders/ui.vert -o shaders/ui.vert.spv
glslc shaders/ui.frag -o shaders/ui.frag.spv
glslc -fshader-stage=vertex shaders/shadow.vert.glsl -o shaders/shadow.vert.spv

glslc --target-env=vulkan1.2 shaders/raygen.rgen -o shaders/raygen.spv
glslc --target-env=vulkan1.2 shaders/miss.rmiss -o shaders/miss.spv
glslc --target-env=vulkan1.2 shaders/shadow.rmiss -o shaders/shadow.rmiss.spv
glslc --target-env=vulkan1.2 shaders/closesthit.rchit -o shaders/closesthit.spv

echo "=== [2/3] COMPILARE C++ ==="
cmake --build build

echo "=== [3/3] RULARE ENGINE ==="
./build/RPGMaker3D
