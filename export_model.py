import os

from data import load_merl_as_tensor
from render import IDX

import torch
from models import DeepBRDFVAE, ColorWiseVAE
import torch.onnx
import numpy as np

print("start")


class Decoder(DeepBRDFVAE):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(n_slices=63, z_dim=12)

    def forward(self, x):
        return self._decode(x)


class Decoder2(ColorWiseVAE):
    def __init__(self, z_dim):
        super(Decoder2, self).__init__(z_dim=z_dim)

    def forward(self, x):
        return self._decode(x)


NAME = "reg_z12"

model = Decoder2(int(NAME.split("_z")[-1])) if "color" in NAME else Decoder()
model.load_state_dict(
    torch.load(f"chckpnt_fin/{NAME}/model_end.pth", map_location=torch.device("cpu"))
)

model.eval()
print("finished loading model")

folder = "BRDFDatabase/brdfs"
mat_name = tuple(
    os.path.join(folder, file)
    for file in os.listdir(folder)
    if file.endswith(".binary") and not file.startswith(".")
)[0]

print("init load", mat_name)


brdf = torch.unsqueeze(load_merl_as_tensor(mat_name), 0)[:, IDX]
enc = model._encode(brdf)

latent = enc[:, : model.z_dim]

PATH = f"www/assets/weights/{NAME}.onnx"

z = torch.onnx.export(
    model,
    latent,
    PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
# print(z)
# quit()

with torch.no_grad():
    out = model(latent)
print(out.shape)


import onnx

onnx_model = onnx.load(PATH)
print(onnx.checker.check_model(onnx_model))

import onnxruntime

ort_session = onnxruntime.InferenceSession(PATH)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(latent)}
ort_outs = ort_session.run(["output"], ort_inputs)
print(ort_outs[0].shape)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-02, atol=1e-02)


print("Exported model has been tested with ONNXRuntime, and the result looks good!")

quit()
