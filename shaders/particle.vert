#version 460

layout(location = 0) in vec3 inPosition; // world-space billboard corner (CPU-expanded)
layout(location = 1) in vec2 inUV;       // [0,1] within the quad
layout(location = 2) in vec4 inColor;

// Only view + proj needed from the UBO — the rest of the struct is ignored.
layout(binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(inPosition, 1.0);
    fragUV      = inUV;
    fragColor   = inColor;
}
