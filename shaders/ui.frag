#version 460

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

// Font atlas (R8_UNORM, 128x64, 16 chars/row × 8 rows, each char 8x8 pixels)
layout(set = 0, binding = 0) uniform sampler2D fontAtlas;

layout(push_constant) uniform UIPushConst {
    int mode; // 0 = solid color, 1 = font atlas alpha mask
} push;

void main() {
    if (push.mode == 1) {
        // Sample alpha from the font atlas R channel; vertex color is the text tint
        float alpha = texture(fontAtlas, fragUV).r;
        if (alpha < 0.05) discard;
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
    } else {
        // Solid rectangle — just output vertex color directly
        outColor = fragColor;
    }
}
