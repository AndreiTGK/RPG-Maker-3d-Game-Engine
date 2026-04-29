#version 460

layout(location = 0) out vec2 fragUV;

// Full-screen triangle from vertex index — no vertex buffer needed.
// Vertices at: (-1,-1), (3,-1), (-1,3) cover the entire clip-space screen.
void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV      = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
