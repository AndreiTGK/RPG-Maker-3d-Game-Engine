#version 460

layout(location = 0) in vec3  inPosition;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inTexCoord;
layout(location = 3) in uvec4 inJoints;
layout(location = 4) in vec4  inWeights;

struct GpuPointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float radius;
};

layout(set = 0, binding = 0) uniform UniformBufferObject {
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

// Per-entity SSBO at set=1, binding=0: joint matrices already premultiplied by IBM.
layout(set = 1, binding = 0) readonly buffer JointMatrices {
    mat4 joints[];
};

layout(push_constant) uniform Push {
    mat4  modelMatrix;
    int   textureIndex;
    float metallic;
    float roughness;
} push;

layout(location = 0) out vec3      fragColor;
layout(location = 1) out vec2      fragTexCoord;
layout(location = 2) out vec3      fragPosWorld;
layout(location = 3) out vec3      fragNormal;
layout(location = 4) out flat int  fragTexIndex;

void main() {
    // Blend joint matrices weighted by vertex weights
    mat4 skinMat =
        inWeights.x * joints[inJoints.x] +
        inWeights.y * joints[inJoints.y] +
        inWeights.z * joints[inJoints.z] +
        inWeights.w * joints[inJoints.w];

    vec4 skinnedPos = skinMat * vec4(inPosition, 1.0);
    vec4 worldPos   = push.modelMatrix * skinnedPos;
    gl_Position     = ubo.proj * ubo.view * worldPos;

    fragColor    = vec3(1.0);
    fragTexCoord = inTexCoord;
    fragPosWorld = worldPos.xyz;

    // Transform normal through skinning rotation + model rotation
    mat3 skinRot    = mat3(skinMat);
    mat3 normalMat  = transpose(inverse(mat3(push.modelMatrix) * skinRot));
    fragNormal      = normalMat * inNormal;

    fragTexIndex = push.textureIndex;
}
