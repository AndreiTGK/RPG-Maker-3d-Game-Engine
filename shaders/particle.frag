#version 460

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // Radial alpha: UV [0,1] → centered [-1,1], fade out toward edges.
    vec2  c    = fragUV * 2.0 - 1.0;
    float dist = length(c);
    float alpha = max(0.0, 1.0 - dist);
    alpha = alpha * alpha; // soften edge

    outColor = vec4(fragColor.rgb, fragColor.a * alpha);
}
