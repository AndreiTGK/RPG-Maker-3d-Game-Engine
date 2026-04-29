#version 460

// UIVertex layout — matches struct UIVertex in VulkanEngine.hpp
layout(location = 0) in vec2 inPos;    // NDC [-1,1]
layout(location = 1) in vec2 inUV;     // texture / font atlas UVs [0,1]
layout(location = 2) in vec4 inColor;  // RGBA tint

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0);
    fragUV    = inUV;
    fragColor = inColor;
}
