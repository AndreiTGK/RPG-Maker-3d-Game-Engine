#version 460

layout(location = 0) in vec3 fragWorldPos;

layout(push_constant) uniform Push {
    mat4 modelMatrix;
    mat4 faceViewProj;
    vec4 lightPosRadius;
} push;

layout(location = 0) out float outLinearDepth;

void main() {
    vec3  lightPos = push.lightPosRadius.xyz;
    float radius   = max(push.lightPosRadius.w, 0.001);
    outLinearDepth = length(fragWorldPos - lightPos) / radius;
}
