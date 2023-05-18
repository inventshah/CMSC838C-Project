#version 300 es
precision highp float;
precision mediump sampler3D;

uniform vec3 incidentVector;
uniform sampler3D data;
uniform samplerCube envCube;

uniform int passNumber;
uniform int stepSize;
uniform int totalSamples;

uniform sampler2D probTex;
uniform sampler2D marginalProbTex;
uniform vec2 texDims;

uniform bool useImportanceSampling;

in vec4 worldSpaceVert;
in vec3 eyeSpaceNormal;
in vec3 eyeSpaceTangent;
in vec3 eyeSpaceBitangent;

out vec4 color;

const float M_PI = 3.1415926535897932384626433832795;

const mat4x4 envRotMatrixInverse = mat4(1);

vec3 esNormal, esTangent, esBitangent, viewVec;
mat3 LocalToWorld, WorldToLocal;
vec3 tsSampleDir, tsViewVec;

int phi_diff_index(float phi_diff) {
	if(phi_diff < 0.0)
		phi_diff += M_PI;

	return clamp(int(phi_diff * (1.0 / M_PI * 180.0)), 0, 180 - 1);
}

int theta_half_index(float theta_half) {
	if(theta_half <= 0.0)
		return 0;

	return clamp(int(sqrt(theta_half * (2.0 / M_PI)) * 90.), 0, 90 - 1);
}

int theta_diff_index(float theta_diff) {
	return clamp(int(theta_diff * (2.0 / M_PI * 90.)), 0, 90 - 1);
}

vec3 BRDF(vec3 toLight, vec3 toViewer, vec3 normal, vec3 tangent, vec3 bitangent) {
	vec3 H = normalize(toLight + toViewer);
	float theta_H = acos(clamp(dot(normal, H), 0., 1.));
	float theta_diff = acos(clamp(dot(H, toLight), 0., 1.));
	float phi_diff = 0.;

	if(theta_diff < 1e-3) {
					// phi_diff indeterminate, use phi_half instead
		phi_diff = atan(clamp(-dot(toLight, bitangent), -1., 1.), clamp(dot(toLight, tangent), -1., 1.));
	} else if(theta_H > 1e-3) {
					// use Gram-Schmidt orthonormalization to find diff basis vectors
		vec3 u = -normalize(normal - dot(normal, H) * H);
		vec3 v = cross(H, u);
		phi_diff = atan(clamp(dot(toLight, v), -1., 1.), clamp(dot(toLight, u), -1., 1.));
	} else
		theta_H = 0.;

	int size = textureSize(data, 0).x / 3;
	int f = 180 / (size - 1);

	// first slice indices and rgb values
	int ind = phi_diff_index(phi_diff) / f;
	int ind1 = theta_diff_index(theta_diff);
	int ind2 = theta_half_index(theta_H);

	float r1 = texelFetch(data, ivec3(ind, ind1, ind2), 0).r;
	float g1 = texelFetch(data, ivec3(ind + size, ind1, ind2), 0).r;
	float b1 = texelFetch(data, ivec3(ind + size * 2, ind1, ind2), 0).r;

	// next slice rgb values
	ind = (ind + 1) % size;
	float r2 = texelFetch(data, ivec3(ind, ind1, ind2), 0).r;
	float g2 = texelFetch(data, ivec3(ind + size, ind1, ind2), 0).r;
	float b2 = texelFetch(data, ivec3(ind + size * 2, ind1, ind2), 0).r;

	// mix between phi_diff and phi_diff + 1
	float rem = float(phi_diff_index(phi_diff) % f);
	return (vec3(r1, g1, b1) * (float(f) - rem) + vec3(r2, g2, b2) * rem) / float(f);
}

vec3 computeWithDirectionalLight(vec3 surfPt, vec3 incidentVector, vec3 viewVec, vec3 normal, vec3 tangent, vec3 bitangent) {
	vec3 b = max(BRDF(incidentVector, viewVec, normal, tangent, bitangent), vec3(0.0));
	b *= dot(normal, incidentVector);
	return b;
}

uint hash(uint x, uint y) {
	const uint M = 1664525u, C = 1013904223u;
	uint seed = (x * M + y + C) * M;
    // tempering (from Matsumoto)
	seed ^= (seed >> 11u);
	seed ^= (seed << 7u) & 0x9d2c5680u;
	seed ^= (seed << 15u) & 0xefc60000u;
	seed ^= (seed >> 18u);
	return seed;
}

float hammersleySample(uint bits, uint seed) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
	bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
	bits ^= seed;
	return float(bits) * 2.3283064365386963e-10; // divide by 1<<32
}

vec3 uvToVector(vec2 uv) {
	vec3 rv = vec3(0.0);

    // face order is px nx py ny pz nz
	float face = floor(uv.x * 6.0);
	uv.x = uv.x * 6.0 - face;
	uv = uv * 2.0 - vec2(1.0);

    // x
	if(face < 1.5) {
        // s = 1 (for +x) or -1 (for -x)
		float s = sign(0.5 - face);
		rv = vec3(s, uv.y, -s * uv.x);
	}

    // y
	else if(face < 3.5) {
		float s = sign(2.5 - face);
		rv = vec3(uv.x, s, -s * uv.y);
	}

    // z
	else {
		float s = sign(4.5 - face);
		rv = vec3(s * uv.x, uv.y, s);
	}

    // note: vector is not normalized so that the length can be determined later
    // to compute projected solid angle at cube position
	return rv;
}

float warpSample1D(sampler2D tex, float texDim, float u, float v, out float probInv) {
	float invTexDim = 1.0 / texDim;

    // evaluate approximate inverse cdf
    // Note: cvs are at pixel centers with implied end points at (0,0) and (1,1)
    // data[0] corresponds to u = 0.5/texDim
	float uN = u * texDim - 0.5;
	float ui = floor(uN);
	float frac = uN - ui;
    // segment spanning u is data[ui] and data[ui+1]
    // sample texture at texel centers (data[ui] = texture((ui+.5)/texDim))
	float t0 = (ui + .5) * invTexDim, t1 = (ui + 1.5) * invTexDim;

	float cdf0 = t0 < 0. ? // data[-1] is -data[0]  (reflect around (0,0))
	-texture(tex, vec2(.5 * invTexDim, v)).r : texture(tex, vec2(t0, v)).r;
	float cdf1 = t1 > 1. ? // data[texDim] = 2-data[texDim-1]  (reflect around (1,1))
	2. - texture(tex, vec2(1. - .5 * invTexDim, v)).r : texture(tex, vec2(t1, v)).r;

    // infer 1/pdf from slope of inverse cdf
	probInv = (texDim * (cdf1 - cdf0));

    // linear interp cdf values to get warped sample
	float uPrime = cdf0 + frac * (cdf1 - cdf0);

	return uPrime;
}

vec2 warpSample(vec2 uv, out float probInv) {
	float uProbInv, vProbInv;
	float vPrime = warpSample1D(marginalProbTex, texDims.y, uv.y, 0.5, vProbInv);
	float uPrime = warpSample1D(probTex, texDims.x, uv.x, vPrime, uProbInv);

	probInv = uProbInv * vProbInv;
	return vec2(uPrime, vPrime);
}

vec4 envMapSample(float u, float v) {
	float probInv = 1.;
	vec2 uv = vec2(u, v);

	if(useImportanceSampling)
		uv = warpSample(uv, probInv); // will overwrite prob with actual pdf value

	vec3 esSampleDir = uvToVector(uv);
    // TODO - precompute LocalToWorld * envRotMatrixInverse
	vec3 tsSampleDir = normalize(LocalToWorld * mat3(envRotMatrixInverse) * esSampleDir);

    // cosine weight
	float cosine = max(0.0, dot(tsSampleDir, vec3(0, 0, 1)));
	if(cosine <= 0.)
		return vec4(vec3(0.), 1.0);

    // since we're working in tangent space, the basis vectors can be nice and easy and hardcoded
	vec3 brdf = max(BRDF(tsSampleDir, tsViewVec, vec3(0, 0, 1), vec3(1, 0, 0), vec3(0, 1, 0)), vec3(0.0));

    // sample env map
	vec3 envSample = textureLod(envCube, esSampleDir, 0.0).rgb;

    // dA (area of cube) = (6*2*2)/N  (Note: divide by N happens later)
    // dw = dA / r^3 = 24 * pow(x*x + y*y + z*z, -1.5) (see pbrt v2 p 947).
	float dw = 24. * pow(esSampleDir.x * esSampleDir.x +
		esSampleDir.y * esSampleDir.y +
		esSampleDir.z * esSampleDir.z, -1.5);

	vec3 result = envSample * brdf * (probInv * cosine * dw);

    // hack - clamp outliers to eliminate hot spot samples
	if(useImportanceSampling)
		result = min(result, vec3(50.));

	return vec4(result, 1.0);
}

vec4 computeIBL() {
	vec4 result = vec4(0.0);
	uint seed1 = hash(uint(gl_FragCoord.x), uint(gl_FragCoord.y));
	uint seed2 = hash(seed1, 1000u);

	float inv = 1.0 / float(totalSamples);
	float bigInv = inv * float(stepSize);

	float u = float(seed1) * 2.3283064365386963e-10;

	for(int i = passNumber; i < totalSamples; i += stepSize) {
		float uu = fract(u + float(i) * inv);
		float vv = fract(hammersleySample(uint(i), seed2));

		result += envMapSample(uu, vv);
	}

	return result;
}

void main() {
	// esNormal = normalize(worldSpaceVert.xyz);
	// esTangent = normalize(cross(vec3(0., 1., 0.), esNormal));
	// esBitangent = normalize(cross(esNormal, esTangent));

	esNormal = normalize(eyeSpaceNormal);
	esTangent = normalize(eyeSpaceTangent);
	esBitangent = normalize(eyeSpaceBitangent);

    //viewVec = -normalize( eyeSpaceVert );
	viewVec = vec3(0, 0, 1);

	WorldToLocal = mat3(esTangent, esBitangent, esNormal);
	LocalToWorld = transpose(WorldToLocal);

    // envSampleRotMatrix = mat3(envRotMatrix) * WorldToLocal;

	tsSampleDir = normalize(incidentVector);
	tsViewVec = LocalToWorld * viewVec;

	vec4 ibl = computeIBL();
	// vec3 b = ibl.rgb / ibl.a;
	// b = pow(b, vec3(1.0 / 2.2));

	// color = vec4(clamp(b, vec3(0.), vec3(1.)), 1.);
	color = ibl;
}