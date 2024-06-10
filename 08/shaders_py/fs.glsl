#version 330 core

in vec4 vertexColour;
out vec4 color;

void main () {
    color = vec4(
        abs(vertexColour.x),
        0.0,
        0.0,
        1.0,
    );
}