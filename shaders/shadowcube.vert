#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(push_constant) uniform Push {
    mat4 modelMatrix;
    mat4 faceViewProj;
    vec4 lightPosRadius; // xyz = world light pos, w = light radius
} push;

layout(location = 0) out vec3 fragWorldPos;

void main() {
    vec4 world   = push.modelMatrix * vec4(inPosition, 1.0);
    fragWorldPos = world.xyz;
    gl_Position  = push.faceViewProj * world;
}
