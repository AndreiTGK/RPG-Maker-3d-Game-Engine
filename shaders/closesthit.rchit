#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT bool isShadowed;

hitAttributeEXT vec2 attribs;

layout(binding = 0) uniform accelerationStructureEXT topLevelAS;

struct GpuPointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float radius;
};

layout(binding = 2) uniform UniformBufferObject {
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

layout(binding = 3, std430) readonly buffer VertexBuffers { float v[]; } vbufs[];
layout(binding = 4, std430) readonly buffer IndexBuffers  { uint  i[]; } ibufs[];

layout(binding = 5) uniform sampler2D texSamplers[];

struct Vertex {
    vec3 pos;
    vec3 color;
    vec3 normal;
    vec2 texCoord;
};

Vertex unpackVertex(uint meshID, uint index) {
    uint offset = index * 11;
    Vertex v;
    v.pos      = vec3(vbufs[nonuniformEXT(meshID)].v[offset],    vbufs[nonuniformEXT(meshID)].v[offset+1], vbufs[nonuniformEXT(meshID)].v[offset+2]);
    v.color    = vec3(vbufs[nonuniformEXT(meshID)].v[offset+3],  vbufs[nonuniformEXT(meshID)].v[offset+4], vbufs[nonuniformEXT(meshID)].v[offset+5]);
    v.normal   = vec3(vbufs[nonuniformEXT(meshID)].v[offset+6],  vbufs[nonuniformEXT(meshID)].v[offset+7], vbufs[nonuniformEXT(meshID)].v[offset+8]);
    v.texCoord = vec2(vbufs[nonuniformEXT(meshID)].v[offset+9],  vbufs[nonuniformEXT(meshID)].v[offset+10]);
    return v;
}

void main() {
    uint meshID = gl_InstanceCustomIndexEXT >> 16;
    uint texID  = gl_InstanceCustomIndexEXT & 0xFFFF;

    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    uint i0 = ibufs[nonuniformEXT(meshID)].i[gl_PrimitiveID * 3 + 0];
    uint i1 = ibufs[nonuniformEXT(meshID)].i[gl_PrimitiveID * 3 + 1];
    uint i2 = ibufs[nonuniformEXT(meshID)].i[gl_PrimitiveID * 3 + 2];

    Vertex v0 = unpackVertex(meshID, i0);
    Vertex v1 = unpackVertex(meshID, i1);
    Vertex v2 = unpackVertex(meshID, i2);

    vec3 pos    = v0.pos      * barycentrics.x + v1.pos      * barycentrics.y + v2.pos      * barycentrics.z;
    vec3 normal = v0.normal   * barycentrics.x + v1.normal   * barycentrics.y + v2.normal   * barycentrics.z;
    vec2 uv     = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;

    if (length(normal) < 0.01) {
        normal = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
    } else {
        normal = normalize(normal);
    }

    vec3 worldPos    = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));
    vec3 worldNormal = normalize(vec3(gl_ObjectToWorldEXT * vec4(normal, 0.0)));

    if (dot(worldNormal, gl_WorldRayDirectionEXT) > 0.0) {
        worldNormal = -worldNormal;
    }

    // Texture / base color
    vec3 baseColor = texture(texSamplers[nonuniformEXT(texID)], uv).rgb;
    if (length(baseColor) < 0.01) {
        baseColor = v0.color * barycentrics.x + v1.color * barycentrics.y + v2.color * barycentrics.z;
        if (length(baseColor) < 0.01) baseColor = vec3(0.8);
    }

    vec3 origin = worldPos + worldNormal * 0.05;
    uint shadowFlags = gl_RayFlagsTerminateOnFirstHitEXT |
                       gl_RayFlagsOpaqueEXT              |
                       gl_RayFlagsSkipClosestHitShaderEXT;

    if (ubo.numActiveLights > 0) {
        // Emissive glow for lights[0]
        if (length(ubo.lights[0].position - worldPos) < 1.5) {
            hitValue = ubo.lights[0].color;
            return;
        }

        vec3 pointTotal = vec3(0.0);
        for (int i = 0; i < ubo.numActiveLights; i++) {
            vec3  lightVec  = ubo.lights[i].position - worldPos;
            float lightDist = length(lightVec);

            if (lightDist > ubo.lights[i].radius) continue;

            vec3  lightDir = normalize(lightVec);
            float diff     = max(dot(worldNormal, lightDir), 0.0);

            float falloff     = clamp(1.0 - (lightDist / ubo.lights[i].radius), 0.0, 1.0);
            falloff           = falloff * falloff;
            float attenuation = falloff * ubo.lights[i].intensity;

            float shadow = 1.0;
            if (diff > 0.0) {
                isShadowed = true;
                float tMax = lightDist - 0.1;
                if (tMax > 0.05) {
                    traceRayEXT(topLevelAS, shadowFlags, 0xFF, 0, 0, 1, origin, 0.05, lightDir, tMax, 1);
                } else {
                    isShadowed = false;
                }
                if (isShadowed) shadow = 0.0;
            }

            pointTotal += diff * ubo.lights[i].color * attenuation * shadow * baseColor;
        }

        // Sun contributes soft ambient diffuse (no shadow ray in multi-light mode)
        vec3  sunDir  = normalize(ubo.sunDirection);
        float sunDiff = max(dot(worldNormal, -sunDir), 0.0);
        vec3  sunContrib = sunDiff * vec3(1.0, 0.95, 0.9) * 0.3 * baseColor;

        hitValue = baseColor * ubo.ambientLight + sunContrib + pointTotal;
    } else {
        // Sun-only mode
        vec3  sunDir  = normalize(ubo.sunDirection);
        float sunDiff = max(dot(worldNormal, -sunDir), 0.0);
        float sunShadow = 1.0;

        if (sunDiff > 0.0) {
            isShadowed = true;
            traceRayEXT(topLevelAS, shadowFlags, 0xFF, 0, 0, 1, origin, 0.05, -sunDir, 10000.0, 1);
            if (isShadowed) sunShadow = 0.0;
        }

        vec3 sunContrib = sunDiff * sunShadow * vec3(1.0, 0.95, 0.9) * 0.8;
        hitValue = baseColor * ubo.ambientLight + sunContrib * baseColor;
    }
}
