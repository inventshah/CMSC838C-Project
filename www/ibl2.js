"use strict";

window.start_x = 0;
window.start_y = 0;
window.half_width = false;

async function load_all(uris, load_func, prefix) {
	return Object.assign(
		{},
		...(await Promise.all(
			uris.map(async (file) => ({
				[file]: await load_func(`${prefix}/${file}`),
			}))
		))
	);
}

async function load_text(uri) {
	return await fetch(uri).then((r) => r.text());
}

async function load_texture_binary(uri) {
	const bytes = await fetch(uri)
		.then((r) => r.blob())
		.then((b) => b.arrayBuffer())
		.catch(console.error);
	const [width, height] = new Int32Array(bytes, 0, 2);
	const data = new Float32Array(bytes, 8);

	return { width, height, data };
}

async function load_environment(uri) {
	const textures = await Promise.all(
		["im1", "im2", "im3", "im4", "im5", "im6", "prob", "marginalProb"].map(
			async (file) => ({
				[file]: await load_texture_binary(`${uri}/${file}.tex`),
			})
		)
	);

	return Object.assign({}, ...textures);
}

function extents(data) {
	let xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = ymin = zmin = Number.MAX_VALUE;
	xmax = ymax = zmax = Number.MIN_VALUE;
	for (let i = 0; i < data.length; ++i) {
		const [x, y, z] = data[i];
		xmin = Math.min(xmin, +x);
		xmax = Math.max(xmax, +x);
		ymin = Math.min(ymin, +y);
		ymax = Math.max(ymax, +y);
		zmin = Math.min(zmin, +z);
		zmax = Math.max(zmax, +z);
	}
	return { xmin, ymin, zmin, xmax, ymax, zmax };
}

// super basic Wavefront OBJ loader
async function load_obj(uri) {
	const text = await load_text(uri);

	const vertices = [];
	const normals = [];

	const model = {
		position: { numComponents: 3, data: [] },
		normal: { numComponents: 3, data: [] },
	};

	for (const line of text.split("\n")) {
		if (!line || line[0] == "#") continue;

		const [prefix, ...pt] = line.split(" ");
		if (prefix == "v") vertices.push(pt);
		else if (prefix == "vt") {
			/* ignore */
		} else if (prefix == "vn") normals.push(pt);
		else if (prefix == "f") {
			if (pt.length != 3) continue;
			for (const part of pt) {
				const [vface, _tface, nface] = part
					.split("/")
					.map((a) => +a - 1);

				for (const item of vertices[vface])
					model.position.data.push(+item);
				for (const item of normals[nface])
					model.normal.data.push(+item);
			}
		}
	}

	const { xmin, ymin, zmin, xmax, ymax, zmax } = extents(vertices);

	const xcenter = xmin + (xmax - xmin) * 0.5;
	const ycenter = ymin + (ymax - ymin) * 0.5;
	const zcenter = zmin + (zmax - zmin) * 0.5;

	const scale = Math.min(
		2 / (xmax - xmin),
		2 / (ymax - ymin),
		2 / (zmax - zmin)
	);

	for (let i = 0; i < model.position.data.length; i += 3) {
		model.position.data[i + 0] =
			(model.position.data[i + 0] - xcenter) * scale;
		model.position.data[i + 1] =
			(model.position.data[i + 1] - ycenter) * scale;
		model.position.data[i + 2] =
			(model.position.data[i + 2] - zcenter) * scale;
	}

	return model;
}

// from Observable
async function load_cubeCross(uri, size = 256) {
	return await new Promise((resolve) => {
		const im = new Image();
		im.crossOrigin = "anonymous";
		im.src = uri;
		im.onload = () => {
			resolve(
				[
					[size * 2, size],
					[0, size],
					[size, 0],
					[size, size * 2],
					[size, size],
					[size * 3, size],
				].map(([x, y]) => {
					const img = document.createElement("canvas");
					img.width = img.height = size;
					const ctx = img.getContext("2d");
					ctx.drawImage(im, x, y, size, size, 0, 0, size, size);
					return { img, size };
				})
			);
		};
	});
}

// preloads all WebGL related assets
async function load_assets() {
	const PREFIX = "assets";
	const shaders = await load_all(
		["base.fs", "base.vs", "comp.fs", "ibl_final.fs", "quad.vs"],
		load_text,
		`${PREFIX}/shaders`
	);

	const objects = await load_all(["teapot.obj"], load_obj, PREFIX);

	const envs = await load_all(
		["beach", "spot"],
		load_environment,
		`${PREFIX}/env`
	);

	const cubemaps = await load_all(
		["matpreview.png"],
		load_cubeCross,
		`${PREFIX}/env`
	);

	return { shaders, objects, envs, cubemaps };
}

function init_webgl(canvas, brdf_id) {
	const gl = canvas.getContext("webgl2");
	gl.getExtension("EXT_float_blend");
	gl.getExtension("EXT_color_buffer_float");
	gl.getExtension("OES_texture_float_linear");
	gl.getExtension("WEBGL_multi_draw");

	// create buffer infos
	const quadInfo = twgl.primitives.createXYQuadBufferInfo(gl);
	const sphereInfo = twgl.primitives.createSphereBufferInfo(gl, 1, 100, 100);
	const teapotInfo = twgl.createBufferInfoFromArrays(
		gl,
		assets.objects["teapot.obj"]
	);

	const iblProgramInfo = create_program(gl, "base.vs", "base.fs");
	const compProgramInfo = create_program(gl, "quad.vs", "comp.fs");
	const finalProgramInfo = create_program(gl, "quad.vs", "ibl_final.fs");

	const beachEnvironment = environment_to_textures(gl, "beach");
	const spotEnvironment = environment_to_textures(gl, "spot");
	const matpreviewCube = twgl.createTexture(gl, {
		target: gl.TEXTURE_CUBE_MAP,
		flipY: false,
		src: assets.cubemaps["matpreview.png"].map((d) => d.img),
	});

	let passNumber = 0;
	let stepSize = 271;
	let totalSamples = stepSize * 15;

	const comp_fbo = new_fbo(gl, 1250, 1250);
	const brdf_fbo = new_fbo(gl, 1250, 1250);

	const incidentVector = [0, 0, 1];
	const projectionMatrix = twgl.m4.ortho(-1.1, 1.1, -1.1, 1.1, 0.01, 50);

	function render(step) {
		let lookTheta = +document.getElementById("theta").value;
		let lookPhi = +document.getElementById("phi").value;

		const modelViewMatrix = twgl.m4.inverse(
			twgl.m4.scale(
				twgl.m4.lookAt(
					[
						Math.sin(lookTheta) * Math.cos(lookPhi) * 25,
						Math.cos(lookTheta) * 25,
						Math.sin(lookTheta) * Math.sin(lookPhi) * 25,
					],
					[0, 0, 0],
					[0, 1, 0]
				),
				[1, 1, 1]
			)
		);

		const environment = document.getElementById("envMap").checked
			? beachEnvironment
			: spotEnvironment;

		const envCube = environment.envCube;

		if (window[brdf_id]) {
			if (passNumber < stepSize) passNumber++;
			else passNumber = 0;

			const modelInfo = document.getElementById("model").checked
				? teapotInfo
				: sphereInfo;

			const uniforms = {
				data: window[brdf_id],
				texDims: [256 * 6, 256], // TODO: set from environment
				useImportanceSampling:
					document.getElementById("ibl_is").checked,
				...environment,
				envCube,
				incidentVector,
				projectionMatrix,
				modelViewMatrix,
				passNumber,
				totalSamples,
				stepSize,
			};

			// render single frame
			gl.enable(gl.DEPTH_TEST);
			gl.bindFramebuffer(gl.FRAMEBUFFER, brdf_fbo.id);
			gl.useProgram(iblProgramInfo.program);

			twgl.setBuffersAndAttributes(gl, iblProgramInfo, modelInfo);
			gl.clear(gl.DEPTH_BUFFER_BIT | gl.COLOR_BUFFER_BIT);

			gl.viewport(0, 0, brdf_fbo.width, brdf_fbo.height);
			twgl.setUniforms(iblProgramInfo, uniforms);
			twgl.drawBufferInfo(gl, modelInfo);
			gl.disable(gl.DEPTH_TEST);

			// combine
			gl.enable(gl.DEPTH_TEST);
			gl.bindFramebuffer(gl.FRAMEBUFFER, comp_fbo.id);
			gl.useProgram(compProgramInfo.program);

			twgl.setBuffersAndAttributes(gl, compProgramInfo, quadInfo);
			gl.clear(gl.DEPTH_BUFFER_BIT);
			gl.enable(gl.BLEND);
			gl.blendFunc(gl.ONE, gl.ONE);

			gl.viewport(0, 0, comp_fbo.width, comp_fbo.height);
			twgl.setUniforms(compProgramInfo, {
				...uniforms,
				result: brdf_fbo.color,
			});
			twgl.drawBufferInfo(gl, quadInfo);
			gl.disable(gl.BLEND);
			gl.disable(gl.DEPTH_TEST);

			// render to canvas
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
			gl.useProgram(finalProgramInfo.program);

			twgl.setBuffersAndAttributes(gl, finalProgramInfo, quadInfo);
			gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
			gl.clear(gl.DEPTH_BUFFER_BIT);
			twgl.setUniforms(finalProgramInfo, {
				...uniforms,
				result: comp_fbo.color,
			});
			twgl.drawBufferInfo(gl, quadInfo);
		}
		requestAnimationFrame(render);
	}

	function resizeCanvas() {
		const len = Math.min(window.innerWidth, window.innerHeight);
		const size =
			len * (len < 900 ? 0.5 : 0.8) * (window.half_width ? 0.7 : 1);
		canvas.width = size;
		canvas.height = size;
		gl.viewport(0, 0, 1250, 1250);
	}

	resizeCanvas();
	window.addEventListener("resize", resizeCanvas);

	canvas.onmousedown = (evt) => {
		window.start_x = evt.offsetX;
		window.start_y = evt.offsetY;
	};

	const theta = document.getElementById("theta");
	const phi = document.getElementById("phi");
	canvas.onmousemove = (evt) => {
		if (evt.buttons === 0) return;

		const dx = (evt.offsetX - window.start_x) * 0.02;
		const dy = (evt.offsetY - window.start_y) * 0.03;

		phi.value = +phi.value + dx;
		let lookTheta = +theta.value - dy;

		if (lookTheta < 0.001) lookTheta = 0.001;
		if (lookTheta > Math.PI - 0.001) lookTheta = Math.PI - 0.001;

		theta.value = lookTheta;

		window.reset_canvas();

		window.start_x = evt.offsetX;
		window.start_y = evt.offsetY;
	};

	function reset() {
		clear_fbo(gl, comp_fbo);
	}

	return { gl, render, reset, resizeCanvas };
}

(async function () {
	window.assets = await load_assets();

	const instance1 = init_webgl(
		document.getElementById("deep-canvas"),
		"deep-brdf"
	);
	const { run_model } = await load_onnx((data, shape) => {
		brdf_to_texture(instance1.gl, data, shape, "deep-brdf");
		instance1.reset();
	});

	requestAnimationFrame(instance1.render);

	window.reset_canvas = () => {
		instance1.reset();
	};

	document.getElementById("theta").oninput = reset_canvas;
	document.getElementById("phi").oninput = reset_canvas;

	document.getElementById("ibl_is").oninput = reset_canvas;
	document.getElementById("model").oninput = reset_canvas;
	document.getElementById("envMap").oninput = reset_canvas;

	document.getElementById("extra").onchange = () => {
		run_model();
		instance1.reset();
	};
})();

function brdf_to_texture(gl, src, shape, id) {
	const [width, height, depth] = shape;

	const tex = twgl.createTexture(gl, {
		src,
		width,
		height,
		depth,
		target: gl.TEXTURE_3D,
		format: gl.RED,
		internalFormat: gl.R32F,
		type: gl.FLOAT,
		minMag: gl.NEAREST,
	});

	if (window[id]) gl.deleteTexture(window[id]);

	window[id] = tex;
}

const SCALE = [1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0];

function init_brdf_upload(instance, id) {
	const upload = document.getElementById("upload");
	upload.onchange = (e) => {
		const brdf = e.target.files[0];

		brdf.arrayBuffer().then((buffer) => {
			const [A, B, C] = brdf.name.endsWith(".slicedbinary")
				? [90, 90, 21]
				: new Int32Array(buffer, 0, 3);

			const raw = brdf.name.endsWith(".slicedbinary")
				? new Float32Array(buffer)
				: new Float64Array(buffer.slice(12));

			const moved = new Array(A * B * C * 3);
			for (let c = 0; c < 3; c++) {
				for (let i = 0; i < A; i++) {
					for (let j = 0; j < B; j++) {
						for (let k = 0; k < C; k++) {
							moved[i * (B * C * 3) + j * (C * 3) + k + C * c] =
								brdf.name.endsWith(".slicedbinary")
									? undo_log_normalization(
											raw[
												c * (A * B * C) +
													i * (B * C) +
													j * C +
													k
											]
									  ) * SCALE[c]
									: raw[
											c * (A * B * C) +
												i * (B * C) +
												j * C +
												k
									  ] * SCALE[c];
						}
					}
				}
			}

			brdf_to_texture(
				instance.gl,
				new Float32Array(moved),
				[C * 3, A, B],
				id
			);
			instance.reset();
		});
	};
}

function undo_log_normalization(z) {
	z *= Math.log(1.01) - Math.log(0.01);
	z += Math.log(0.01);
	return Math.exp(z) - 0.01;
}

window.latent_vector = new Array(10).fill(0);

async function load_onnx(callback) {
	const session = await ort.InferenceSession.create(
		"assets/weights/og.onnx"
	).catch((e) => alert("the browser was unable to load the neural network"));

	async function execute(vector) {
		const typed = Float32Array.from(vector);
		const tensor = new ort.Tensor("float32", typed, [1, 8]);
		const results = await session.run({ input: tensor });
		return results.output;
	}

	const extra = document.getElementById("extra");

	async function run_model() {
		const M1 = await execute(latent_vector.slice(0, 8));
		const M2 = extra.checked
			? await execute([
					latent_vector[8],
					...latent_vector.slice(1, 7),
					latent_vector[9],
			  ])
			: null;

		const [b, A, B, C] = M1.dims;
		const green = (A / 3) | 0;

		// data[:, i, :, :] = moved[:, :, : i]
		// [i * (B * C) + j * (C) + k] => [j * (C * A) + k * (A) + i]
		const moved = new Array(b * A * B * C);
		for (let i = 0; i < A; i++) {
			for (let j = 0; j < B; j++) {
				for (let k = 0; k < C; k++) {
					moved[j * (C * A) + k * A + i] = undo_log_normalization(
						i < green || i >= 2 * green || !extra.checked
							? M1.data[i * (B * C) + j * C + k]
							: M2.data[(i - green) * (B * C) + j * C + k]
					);
				}
			}
		}

		callback(new Float32Array(moved), [A, B, C]);
	}

	run_model();

	const sliders = document.getElementById("sliders");
	const latentSliders = [];
	for (let i = 0; i < 10; ++i) {
		const labelText = `P${i + 1}`;
		const elm = document.createElement("input");

		elm.setAttribute("type", "range");
		elm.setAttribute("min", "-10");
		elm.setAttribute("max", "10");
		elm.setAttribute("step", "0.01");
		elm.setAttribute("value", "0");
		elm.setAttribute("name", labelText);
		elm.setAttribute("id", labelText);
		const label = document.createElement("label");
		label.innerText = labelText;
		label.setAttribute("for", labelText);
		sliders.appendChild(label);
		sliders.appendChild(elm);

		elm.onchange = (evt) => {
			const v = evt.target.value;
			window.latent_vector[i] = +v;
			run_model();
		};
		latentSliders.push(elm);
	}

	const merlSelector = document.getElementById("merl");
	merlSelector.onchange = (evt) => {
		const matName = evt.target.value;
		if (matName === "none") return;
		const mat = MERL[matName];
		window.latent_vector = [...mat.z1, mat.z2[0] || 0, mat.z2[7] || 0];

		for (let i = 0; i < 10; ++i)
			latentSliders[i].value = window.latent_vector[i];

		extra.checked = mat.z2.length > 0;
		run_model();
	};

	return { run_model };
}
