#version 330 core

precision highp float;

out vec4 color;

in vec4 vertexColour;

void main () {
    // color = vertexColour;

    color = vec4(
        vertexColour.x,
        0.0,
        0.0,
        1.0
    );

}