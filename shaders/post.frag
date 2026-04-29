#version 460

layout(binding = 0) uniform sampler2D hdrBuffer;

layout(push_constant) uniform PostPush {
    float exposure;       // scene exposure multiplier (default 1.0)
    float bloomThreshold; // luminance above which bloom is gathered (default 1.0)
    float bloomStrength;  // bloom mix weight (default 0.05)
} push;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// ACES filmic tone-mapping (Krzysztof Narkowicz approximation)
vec3 acesFilmic(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdrBuffer, fragUV).rgb * push.exposure;

    // Simple single-pass bloom: gather neighbouring bright pixels with
    // a 5×5 separable-like kernel, accumulate only above threshold.
    vec2 texelSize = 1.0 / vec2(textureSize(hdrBuffer, 0));
    // Gaussian-ish weights for distances 0..2
    const float w[3] = float[](0.2270270, 0.3162162, 0.0702703);
    vec3 bloom = vec3(0.0);
    float wSum = 0.0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec3 s = texture(hdrBuffer, fragUV + vec2(x, y) * texelSize * 2.0).rgb;
            float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
            if (lum > push.bloomThreshold) {
                float wxy = w[abs(x)] * w[abs(y)];
                bloom += s * wxy;
                wSum  += wxy;
            }
        }
    }
    if (wSum > 0.0) bloom /= wSum;

    hdr += bloom * push.bloomStrength;

    // ACES tonemapping — output is in [0,1] linear.
    // The swapchain is VK_FORMAT_B8G8R8A8_SRGB, so the hardware applies
    // sRGB gamma automatically; we must NOT apply pow(1/2.2) here.
    vec3 mapped = acesFilmic(hdr);

    outColor = vec4(mapped, 1.0);
}
