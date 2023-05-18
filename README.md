# Data Driven BRDF Editing

[Project webpage](https://www.cs.umd.edu/~shah2022/xr/index.html)

Source code for training VAE and web-based demos.

Download data from:

-   MERL: https://cdfg.csail.mit.edu/wojciech/brdfdatabase
-   EPFL-RGL: https://rgl.epfl.ch/materials

Model weights and preprocessed data used in the report are available by contact.

After training a model, use `python export_model.py` to convert the PyTorch weights to the Onnx format.

Web demos are in `www`. Start a server `python -m http.server 8080` and go to `localhost:8080` the first links direct to the demos.
