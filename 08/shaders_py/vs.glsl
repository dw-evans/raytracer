#version 330 core

in vec3 position;

uniform float scale;

out vec4 vertexColour;

void main() {
    gl_Position = vec4(position.xyz * scale, 1.0);

    vertexColour = vec4(
        position.xy, position.z + 0.5,
        1.0
    );



}



