#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;

struct GpuPointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float radius;
};

layout(binding = 2, set = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix[4];
    vec3 ambientLight;
    float _pad0;
    vec3 sunDirection;
    float shadowsEnabled;
    vec3 skyColor;
    int  numActiveLights;
    GpuPointLight lights[4];
} ubo;

void main() {
    hitValue = ubo.skyColor;
}
