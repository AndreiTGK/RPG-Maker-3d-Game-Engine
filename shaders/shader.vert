#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

struct GpuPointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float radius;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix[4];
    vec3  ambientLight;
    float _pad0;
    vec3  sunDirection;
    float shadowsEnabled;
    vec3  skyColor;
    int   numActiveLights;
    GpuPointLight lights[4];
} ubo;

layout(push_constant) uniform Push {
    mat4  modelMatrix;
    int   textureIndex;
    float metallic;
    float roughness;
} push;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragPosWorld;
layout(location = 3) out vec3 fragNormal;
layout(location = 4) out flat int fragTexIndex;

void main() {
    vec4 worldPos = push.modelMatrix * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragPosWorld = worldPos.xyz;

    // Extragem normala reala a modelului si o rotim in functie de transformarea obiectului
    mat3 normalMatrix = transpose(inverse(mat3(push.modelMatrix)));
    fragNormal = normalMatrix * inNormal;

    fragTexIndex = push.textureIndex;
}
