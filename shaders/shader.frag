#version 460
#extension GL_EXT_nonuniform_qualifier : require

const float PI = 3.14159265359;

struct GpuPointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float radius;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix[4]; // per-light; tile i at atlas UV (i%2*0.5, i/2*0.5)
    vec3 ambientLight;
    float _pad0;
    vec3  sunDirection;
    float shadowsEnabled;
    vec3  skyColor;
    int   numActiveLights;
    GpuPointLight lights[4];
} ubo;

layout(binding = 1) uniform sampler2D texSamplers[];
layout(binding = 2) uniform sampler2D shadowMap;       // directional sun shadow (atlas tile 0)
layout(binding = 3) uniform samplerCubeArray shadowCubeMap; // omnidirectional point-light shadows

layout(push_constant) uniform Push {
    mat4  modelMatrix;
    int   textureIndex;
    float metallic;
    float roughness;
} push;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosWorld;
layout(location = 3) in vec3 fragNormal;
layout(location = 4) in flat int fragTexIndex;

layout(location = 0) out vec4 outColor;

// --- PBR: Cook-Torrance BRDF helpers ---

// Trowbridge-Reitz GGX normal distribution
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a    = roughness * roughness;
    float a2   = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float denom = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith's Schlick-GGX geometry function
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Full PBR radiance for one light direction L (normalized, pointing toward light)
// lightRadiance = lightColor * attenuation (pre-multiplied intensity)
vec3 pbrContrib(vec3 N, vec3 V, vec3 L, vec3 lightRadiance,
                vec3 albedo, float metallic, float roughness) {
    float NdL = max(dot(N, L), 0.0);
    if (NdL <= 0.0) return vec3(0.0);

    vec3  H   = normalize(V + L);
    vec3  F0  = mix(vec3(0.04), albedo, metallic);

    float D   = distributionGGX(N, H, roughness);
    float G   = geometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3  specular = D * G * F / max(4.0 * max(dot(N, V), 0.0) * NdL, 0.001);
    vec3  kD       = (1.0 - F) * (1.0 - metallic);

    return (kD * albedo / PI + specular) * lightRadiance * NdL;
}

// PCF 3×3 shadow sampling from a shadow atlas tile.
// projCoords.xy in [0,1] (NDC-remapped, within the tile), projCoords.z for depth comparison.
// atlasOffset: UV origin of the tile (tile i: (i%2 * 0.5, i/2 * 0.5)), tileScale = 0.5.
float sampleShadow(vec3 projCoords, float bias, vec2 atlasOffset) {
    float shadow    = 0.0;
    vec2  texelSize = 1.0 / textureSize(shadowMap, 0); // 1/4096 per texel
    vec2  centerUV  = atlasOffset + projCoords.xy * 0.5;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, centerUV + vec2(x, y) * texelSize).r;
            shadow += (projCoords.z - bias > pcfDepth) ? 0.2 : 1.0;
        }
    return shadow / 9.0;
}

// Unproject fragPosWorld through lightSpaceMatrix[lightIdx] and return NDC coords (xy in [0,1]).
// Returns false in the w component if the fragment is outside the light frustum.
vec4 toLightSpace(int lightIdx) {
    vec4 posLS = ubo.lightSpaceMatrix[lightIdx] * vec4(fragPosWorld, 1.0);
    vec3 proj  = posLS.xyz / posLS.w;
    proj.xy    = proj.xy * 0.5 + 0.5;
    return vec4(proj, float(
        proj.z > 0.0 && proj.z < 1.0 &&
        proj.x > 0.0 && proj.x < 1.0 &&
        proj.y > 0.0 && proj.y < 1.0));
}

void main() {
    // Unlit grid geometry
    if (fragTexIndex == -1) {
        outColor = vec4(fragColor, 1.0);
        return;
    }

    // Surface normal
    vec3 N = fragNormal;
    if (length(N) < 0.01) {
        N = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
    } else {
        N = normalize(N);
    }

    // View direction: extract camera position from view matrix
    // View = [R | -R*eye] so eye = -R^T * t, where t = view[3].xyz
    vec3 camPos = -transpose(mat3(ubo.view)) * vec3(ubo.view[3]);
    vec3 V      = normalize(camPos - fragPosWorld);

    // Albedo
    vec3 albedo = texture(texSamplers[nonuniformEXT(fragTexIndex)], fragTexCoord).rgb;
    if (length(albedo) < 0.01) albedo = vec3(0.8);

    float metallic  = clamp(push.metallic,  0.0, 1.0);
    float roughness = clamp(push.roughness, 0.04, 1.0); // avoid mirror singularity

    vec3 Lo = vec3(0.0); // accumulated outgoing radiance

    if (ubo.numActiveLights > 0) {
        for (int i = 0; i < ubo.numActiveLights; i++) {
            vec3  lightVec = ubo.lights[i].position - fragPosWorld;
            float dist     = length(lightVec);
            if (dist > ubo.lights[i].radius) continue;

            vec3  L        = normalize(lightVec);
            float falloff  = clamp(1.0 - dist / ubo.lights[i].radius, 0.0, 1.0);
            falloff        = falloff * falloff;
            vec3  radiance = ubo.lights[i].color * ubo.lights[i].intensity * falloff;

            float shadow = 1.0;
            if (ubo.shadowsEnabled > 0.5) {
                vec3  lightToFrag = fragPosWorld - ubo.lights[i].position;
                float linearDist  = length(lightToFrag) / max(ubo.lights[i].radius, 0.001);

                // PCF: 20-tap Poisson sphere for soft omnidirectional shadows
                const vec3 sampleDirs[20] = vec3[20](
                    vec3( 1, 1, 1), vec3( 1,-1, 1), vec3(-1,-1, 1), vec3(-1, 1, 1),
                    vec3( 1, 1,-1), vec3( 1,-1,-1), vec3(-1,-1,-1), vec3(-1, 1,-1),
                    vec3( 1, 1, 0), vec3( 1,-1, 0), vec3(-1,-1, 0), vec3(-1, 1, 0),
                    vec3( 1, 0, 1), vec3(-1, 0, 1), vec3( 1, 0,-1), vec3(-1, 0,-1),
                    vec3( 0, 1, 1), vec3( 0,-1, 1), vec3( 0,-1,-1), vec3( 0, 1,-1)
                );
                float bias       = 0.05;
                float diskRadius = 0.05 * (1.0 + linearDist); // wider disk at distance
                float shadowSum  = 0.0;
                for (int s = 0; s < 20; s++) {
                    vec3  sampleDir    = lightToFrag + normalize(sampleDirs[s]) * diskRadius;
                    float storedDepth  = texture(shadowCubeMap, vec4(sampleDir, float(i))).r;
                    shadowSum += (linearDist - bias > storedDepth) ? 0.0 : 1.0;
                }
                shadow = shadowSum / 20.0;
            }

            Lo += pbrContrib(N, V, L, radiance, albedo, metallic, roughness) * shadow;
        }

        // Soft directional sun contribution — shadowed via atlas tile 0
        vec3  sunL    = normalize(-ubo.sunDirection);
        float sunShadow = 1.0;
        if (ubo.shadowsEnabled > 0.5) {
            vec4 ls = toLightSpace(0);
            if (ls.w > 0.5) {
                float bias = max(0.005 * (1.0 - max(dot(N, sunL), 0.0)), 0.001);
                sunShadow = sampleShadow(ls.xyz, bias, vec2(0.0, 0.0));
            }
        }
        Lo += pbrContrib(N, V, sunL, vec3(1.0, 0.95, 0.9) * 0.3, albedo, metallic, roughness) * sunShadow;

    } else {
        // Sun-only mode: shadow from atlas tile 0
        vec3  sunL = normalize(-ubo.sunDirection);
        float bias = max(0.005 * (1.0 - max(dot(N, sunL), 0.0)), 0.001);

        float shadow = 1.0;
        if (ubo.shadowsEnabled > 0.5) {
            vec4 ls = toLightSpace(0);
            if (ls.w > 0.5)
                shadow = sampleShadow(ls.xyz, bias, vec2(0.0, 0.0));
        }

        Lo += pbrContrib(N, V, sunL, vec3(1.0, 0.95, 0.9), albedo, metallic, roughness) * shadow;
    }

    // Ambient — Fresnel-weighted diffuse IBL approximation
    vec3 F0     = mix(vec3(0.04), albedo, metallic);
    vec3 F_amb  = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD_amb = (1.0 - F_amb) * (1.0 - metallic);
    vec3 ambient = kD_amb * albedo * ubo.ambientLight;

    vec3 color = ambient + Lo;

    // Output raw linear HDR — tonemapping and gamma correction happen in the post pass (post.frag).
    outColor = vec4(color, 1.0);
}
