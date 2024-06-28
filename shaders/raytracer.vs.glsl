#version 460 core

in vec3 position;
out vec3 fragPosition;

void main() {
    gl_Position = vec4(position.x,  position.y, position.z, 1.0);
    fragPosition = gl_Position.xyz;
}



