can use 
```shell
# Build the Docker image

docker build -t llm-fine-tune .

# Run the Docker container

docker run --gpus all llm-fine-tune
```


### Software
Base Image

```bash
nvcr.io/nvidia/pytorch:24.04-py3
```

Contents of the PyTorch container

This container image contains the complete source of the version of PyTorch in `/opt/pytorch`. It is prebuilt and installed in the default Python environment (`/usr/local/lib/python3.10/dist-packages/torch`) in the container image.

The container also includes the following:
* ‣ Ubuntu 22.04 including Python 3.10
* ‣ NVIDIA CUDA 12.4
* ‣ NVIDIA cuBLAS 12.4.5.8
* ‣ NVIDIA cuDNN 9.1.0.70
* ‣ NVIDIA NCCL 2.21.5
* ‣ NVIDIA RAPIDS™ 24.02
* ‣ rdma-core 39.0
* ‣ NVIDIA HPC-X 2.18
* ‣ OpenMPI 4.1.4+
* ‣ GDRCopy 2.3
* ‣ TensorBoard 2.9.0
* ‣ Nsight Compute 2024.1.0.13
* ‣ Nsight Systems 2024.2.1.38
* ‣ NVIDIA TensorRT™ 8.6.3
* ‣ Torch-TensorRT 2.3.0a0
* ‣ NVIDIA DALI® 1.36
* ‣ nvImageCodec 0.2.0.7
* ‣ MAGMA 2.6.2
* ‣ JupyterLab 2.3.2 including Jupyter-TensorBoard
* ‣ TransformerEngine 1.5
* ‣ PyTorch quantization wheel 2.1.2


ssh root@66.114.112.70 -p 58691 -i ~/.ssh/id_ed25519

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHaSRFUDVYFr6Y6Zy0zHIgO26Xx/hkgKE8XWjErcap8F vanshika@DESKTOP-T4ORJP6

Packages

```bash
kaggle
kagglehub
hugginface_hub
detectron2==0.6
transformers==4.43.3
datasets==2.19.0
flash-attn==2.6.2
optimi==0.2.1
triton
transformer_engine == 1.5 
```

### Scripts Flow

```bash
python scripts/prepare_dataset.py
```


```bash
python scripts/prepare_dataset_33k.py
```


Main and model train

```bash
torchrun --nproc_per_node=1 main.py configs/stage1/m0.py --out ../output/stage1/m0/update_last.pth
```

Final weights

```bash
python scripts/prepare_gemma2_for_submission.py
```

