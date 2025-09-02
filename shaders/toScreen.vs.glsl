#version 460 core

in vec2 in_pos;  // vertex positions [-1, 1]
in vec2 in_uv;   // texture coordinates [0, 1]

out vec2 uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
