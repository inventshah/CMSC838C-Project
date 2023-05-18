#version 300 es

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

in vec3 position;
out vec4 worldSpaceVert;
out vec4 eyeSpaceVert;

in vec3 normal;
out vec3 eyeSpaceNormal;
out vec3 eyeSpaceTangent;
out vec3 eyeSpaceBitangent;

void computeTangentVectors(vec3 inVec, out vec3 uVec, out vec3 vVec) {
	uVec = abs(inVec.x) < 0.999 ? vec3(1, 0, 0) : vec3(0, 1, 0);
	uVec = normalize(cross(inVec, uVec));
	vVec = normalize(cross(inVec, uVec));
}

void main() {
	// worldSpaceVert = vec4(position, 1);
	// eyeSpaceVert = modelViewMatrix * worldSpaceVert;
	// gl_Position = projectionMatrix * eyeSpaceVert;

	eyeSpaceVert = modelViewMatrix * vec4(position, 1);
	eyeSpaceNormal = transpose(inverse(mat3(modelViewMatrix))) * normal;

	computeTangentVectors(eyeSpaceNormal, eyeSpaceTangent, eyeSpaceBitangent);

	gl_Position = projectionMatrix * eyeSpaceVert;
}