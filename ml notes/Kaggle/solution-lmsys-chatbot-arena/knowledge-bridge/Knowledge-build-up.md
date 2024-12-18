
# Engine
## Distribution in pytorch

- Multi node training with PyTorch DDP, torch.distributed.launch, torchrun and mpirun [video link](https://www.youtube.com/@lambdacloud)
- Distribution pytorch guide: [dist_overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- Intro to **Invited Talk - PyTorch Distributed (DDP, RPC) - By Facebook Research Scientist Shen Li** : [link](https://www.youtube.com/watch?v=3XUG7cjte2U)
- https://www.youtube.com/@NVIDIADeveloper
	- ![](Pasted%20image%2020241218013405.png)
	- ![](Pasted%20image%2020241218013503.png)if u check for A 100 architecture the speeds will get double
	- ![](Pasted%20image%2020241218013627.png)
	- ![](Pasted%20image%2020241218013740.png)
	- ![](Pasted%20image%2020241218013831.png)
	- ![](Pasted%20image%2020241218013846.png)
	- ![](Pasted%20image%2020241218013914.png)
	- ![](Pasted%20image%2020241218013932.png)
	- ![](Pasted%20image%2020241218013953.png)
	- ![](Pasted%20image%2020241218014023.png)
	- ![](Pasted%20image%2020241218014046.png)
	- ![](Pasted%20image%2020241218014118.png)
	- ![](Pasted%20image%2020241218014127.png)
	- ![](Pasted%20image%2020241218014149.png)
	- ![](Pasted%20image%2020241218014200.png) mutiple gpu using hovord
	- https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
	- ![](Pasted%20image%2020241218014321.png)
	- ![](Pasted%20image%2020241218014335.png)
	- ![](Pasted%20image%2020241218014403.png)
	- ![](Pasted%20image%2020241218015237.png)
	- ![](Pasted%20image%2020241218015257.png)
	- ![](Pasted%20image%2020241218015913.png)
	- ![](Pasted%20image%2020241218020051.png)
	- ![](Pasted%20image%2020241218020727.png)![](Pasted%20image%2020241218020654.png)![](Pasted%20image%2020241218020755.png)![](Pasted%20image%2020241218020829.png)![](Pasted%20image%2020241218020911.png)![](Pasted%20image%2020241218020945.png)![](Pasted%20image%2020241218021020.png) ![](Pasted%20image%2020241218021044.png)
	- ![](Pasted%20image%2020241218021106.png)
	- ![](Pasted%20image%2020241218021140.png)![](Pasted%20image%2020241218021152.png)![](Pasted%20image%2020241218021248.png)
	- ![](Pasted%20image%2020241218021411.png)
	- 


## Agrarparse, ArgumentParser 

- https://youtu.be/yEOmDqSKiYs?si=VnmpjB9K-O1PSVl_ ![](Pasted%20image%2020241218032853.png)



# Model

- Transformer package: https://huggingface.co/transformers/v3.0.2/training.html 
- ![](Pasted%20image%2020241218135156.png)![](Pasted%20image%2020241218135213.png)


## Fvcore

- Fvcore.common.param_scheduler: https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#module-fvcore.common.checkpoint ![](Pasted%20image%2020241218151619.png)![](Pasted%20image%2020241218151630.png)


# Accelerate training : **modeling_gemma2_fast.py** 

### Flash Attention
 [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention | PyTorch](https://pytorch.org/blog/flexattention/)

- https://github.com/Dao-AILab/flash-attention
 
- `logit_softcapping`

### transformer engine 

- Improving Hugging Face Training Efficiency Through Packing with Flash Attention [link](https://huggingface.co/blog/packing-with-FA2)
# Faster inference: **modeling_gemma2.py** 

### Flash Attention
 [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention | PyTorch](https://pytorch.org/blog/flexattention/)

- https://github.com/Dao-AILab/flash-attention
 
- `logit_softcapping`
## LightLLM or vLLM

LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance. LightLLM harnesses the strengths of numerous well-regarded open-source implementations, including but not limited to FasterTransformer, TGI, vLLM, and FlashAttention.


vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8.
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

**Performance benchmark**: We include a performance benchmark at the end of [our blog post](https://blog.vllm.ai/2024/09/05/perf-update.html). It compares the performance of vLLM against other LLM serving engines ([TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [SGLang](https://github.com/sgl-project/sglang) and [LMDeploy](https://github.com/InternLM/lmdeploy)). The implementation is under [nightly-benchmarks folder](https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks) and you can [reproduce](https://github.com/vllm-project/vllm/issues/8176) this benchmark using our one-click runnable script.

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including _parallel sampling_, _beam search_, and more
- Tensor parallelism and pipeline parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
- Prefix caching support
- Multi-lora support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral)
- Embedding Models (e.g. E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).
### context_attention_fwd


## LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams. It has the following core features:

### rms_norm

### fused_rotary_emb



## `xformers` -Toolbox to Accelerate Research on Transformers
