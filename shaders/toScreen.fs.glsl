#version 460 core

uniform sampler2D uTexture;
in vec2 uv;
out vec4 fragColor;


void main() {
    fragColor = texture(uTexture, uv);
}

// #version 460 core

// uniform sampler2D uTexture;
// in vec2 uv;
// out vec4 fragColor;

// // Simple Reinhard tone mapping
// vec3 toneMapReinhard(vec3 hdrColor) {
//     return hdrColor / (hdrColor + vec3(1.0));
// }

// // Gamma correction
// vec3 gammaCorrect(vec3 color, float gamma) {
//     return pow(color, vec3(1.0 / gamma));
// }

// void main() {
//     // Fetch HDR color (could be > 1.0)
//     vec3 hdrColor = texture(uTexture, uv).rgb;

//     // Apply tone mapping (Reinhard in this case)
//     vec3 mapped = toneMapReinhard(hdrColor);

//     // Apply gamma correction (to sRGB, gamma ~2.2)
//     vec3 ldrColor = gammaCorrect(mapped, 2.2);

//     fragColor = vec4(ldrColor, 1.0);
// }
