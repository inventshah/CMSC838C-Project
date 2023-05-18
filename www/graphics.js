function new_fbo(gl, width, height) {
	const color = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, color);
	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		gl.RGBA32F,
		width,
		height,
		0,
		gl.RGBA,
		gl.FLOAT,
		null
	);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

	const id = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, id);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		color,
		0
	);

	const rb = gl.createRenderbuffer();
	gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
	gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_STENCIL, width, height);
	gl.framebufferRenderbuffer(
		gl.FRAMEBUFFER,
		gl.DEPTH_STENCIL_ATTACHMENT,
		gl.RENDERBUFFER,
		rb
	);

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	return {
		id,
		color,
		width,
		height,
	};
}

function clear_fbo(gl, fbo) {
	if (!fbo) return;
	gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.id);
	gl.clear(gl.COLOR_BUFFER_BIT);
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function faces_to_envmap(gl, faces) {
	const faceOrder = [
		gl.TEXTURE_CUBE_MAP_POSITIVE_X,
		gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
		gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
		gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
		gl.TEXTURE_CUBE_MAP_POSITIVE_Z,
		gl.TEXTURE_CUBE_MAP_NEGATIVE_Z,
	];
	const tex = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

	for (let i = 0; i < 6; ++i) {
		const { data, width, height } = faces[i];
		gl.texImage2D(
			faceOrder[i],
			0,
			gl.R11F_G11F_B10F,
			width,
			height,
			0,
			gl.RGB,
			gl.FLOAT,
			data
		);
	}
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

	return tex;
}

function create_program(gl, vs, fs) {
	return twgl.createProgramInfo(gl, [assets.shaders[vs], assets.shaders[fs]]);
}

function environment_to_textures(gl, name) {
	const env = assets.envs[name];
	const envCube = faces_to_envmap(gl, [
		env.im1,
		env.im2,
		env.im3,
		env.im4,
		env.im5,
		env.im6,
	]);
	const probTex = twgl.createTexture(gl, {
		target: gl.TEXTURE_2D,
		format: gl.RED,
		internalFormat: gl.R32F,
		type: gl.FLOAT,
		width: env.prob.width,
		height: env.prob.height,
		src: env.prob.data,
	});
	const marginalProbTex = twgl.createTexture(gl, {
		target: gl.TEXTURE_2D,
		format: gl.RED,
		internalFormat: gl.R32F,
		type: gl.FLOAT,
		width: env.marginalProb.width,
		height: env.marginalProb.height,
		src: env.marginalProb.data,
	});

	return { envCube, probTex, marginalProbTex };
}
