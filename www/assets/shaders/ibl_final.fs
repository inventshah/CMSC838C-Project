#version 300 es
precision highp float;

uniform sampler2D result;
uniform samplerCube envCube;

uniform float aspect;

in vec2 uv;

out vec4 color;

void main() {
	vec4 tex = texture(result, uv);

	if(tex.a < 0.25) {
		vec2 texCoord = uv.st;
		texCoord = texCoord * 2.0 - vec2(1.0);
		// texCoord.x *= aspect;

		vec3 dir = normalize(vec3(texCoord.s, texCoord.t, -1.0));
		tex = vec4(texture(envCube, (vec4(dir, 0.0)).xyz).rgb, 1.0);
	}

	tex /= tex.a;

	color = vec4(pow(tex.rgb, vec3(1.0 / 2.2)), 1.0);
}
