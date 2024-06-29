#version 460 core

in vec2 texCoord;
out vec4 color;

uniform ivec2 scale;
uniform sampler2D tex;

void main()
{
    vec2 scaledCoord = texCoord * scale;
    color = texture(tex, scaledCoord);
}