#version 300 es
precision highp float;

uniform sampler2D result;

in vec2 uv;

out vec4 color;

void main() {
	color = texture(result, uv);
}
