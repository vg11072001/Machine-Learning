
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



# data 

## Data manipulation with some processor
- Subclassing in Python
In the context of Python and machine learning libraries like PyTorch, datasets are often represented as classes that subclass a base class like `torch.utils.data.Dataset`. This base class provides a standard interface for accessing data samples using keys.
Here’s a breakdown of the methods mentioned:
1. `__getitem__()`:
This method is required and is used to fetch a data sample given a key (or index).
It should return a single data sample.

2.`__len__()`:
This method is optional but commonly implemented.
It returns the total number of data samples in the dataset.
This is useful for many Sampler implementations and the default options of DataLoader.

3. `__getitems__()`:
This method is optional and is used for fetching multiple data samples at once.
It accepts a list of indices and returns a list of corresponding data samples.
This can be useful for speeding up batched data loading.

## Data loader with collate fn
In the context of PyTorch's DataLoader, the term "collate" refers to the process of combining individual data samples into a batch. The collate_fn is a function that defines how these samples should be combined.

Explanation
When you fetch data samples from a dataset, each sample might be a tuple, a dictionary, or any other type of object. The DataLoader needs to combine these individual samples into a batch, which is typically a list or a tensor. The collate_fn is responsible for this combination.

Default Collation Behavior
By default, PyTorch's DataLoader uses a collation function that works well for most common data types. Here’s what it does:
- Tensors: Stacks tensors along a new dimension.
- Numbers: Stacks numbers into a tensor.
- Strings: Collects strings into a list.
- Dictionaries: Recursively collates the values of dictionaries.
- Lists/Tuples: Recursively collates the elements of lists or tuples.
For example, if your dataset returns samples as tuples of tensors, the default collation will stack these tensors into a batch.

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # Each sample is a dictionary
        return {'image': self.data[idx][0], 'label': self.data[idx][1]}

    def __len__(self):
        return len(self.data)

# Example data
data = [
    (torch.tensor([1, 2, 3]), 0),
    (torch.tensor([4, 5, 6]), 1),
    (torch.tensor([7, 8, 9]), 0),
]

dataset = CustomDataset(data)
```

```python
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'images': images, 'labels': labels}

dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

for batch in dataloader:
    print(batch)
```
- https://huggingface.co/docs/transformers/main/en/chat_templating
- https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate
- https://huggingface.co/docs/transformers/en/main_classes/data_collator

torch.utils.data
At the heart of PyTorch data loading utility is the torch.utils.data.DataLoader class. 
It represents a Python iterable over a dataset, with support for
1. map-style and iterable-style datasets,
2. customizing data loading order,
3. automatic batching,
4. single- and multi-process data loading,
5. automatic memory pinning.

## Lazy Config from detectron2:
- https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=S36RbV7uk1L8
	- use the sample following code to test Lazy config feature working
```python
%%writefile config.py
# config.py
from detectron2.config import LazyCall as L

# my_app/optimizer.py
class Optimizer:
    def __init__(self, lr, algo):
        self.lr = lr
        self.algo = algo

# my_app/trainer.py
class Trainer:
    def __init__(self, optimizer):
        self.optimizer = optimizer

# Define the configuration
cfg = L(Trainer)(
    optimizer=L(Optimizer)(
        lr=0.01,
        algo="SGD"
    )
)

```

```python
# my_code.py
from detectron2.config import LazyConfig, instantiate

# Load the configuration
cfg = LazyConfig.load("config.py")  # Use relative path

# Instantiate the Trainer object
trainer = instantiate(cfg)

# Access and print the optimizer's learning rate
print(trainer.cfg.optimizer.lr)  # Output: 0.01
```
- further reading https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/lazyconfigs.md

## Tokenizer from transformer

we are using Autotokenizer: 
- https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#autotokenizer

```python
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer_testing = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# Example text
text_testing = "This is an example sentence model path when have acces to dowload weights from HF"
max_length1 = 10
# tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True).input_ids

# Tokenize the text
inputs_testing = tokenizer_testing(text_testing, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs_testing1 = tokenizer_testing(text_testing, add_special_tokens=False, max_length=max_length1, truncation=True)

# Print the tokenized input
print(inputs_testing)
print(inputs_testing1)
print(len(text_testing))
print(len(inputs_testing1.input_ids))
# print(len(inputs_testing['input_ids'][0]))
# print(len(inputs_testing['attention_mask']))
```

for `input_testing` : Return Type: PyTorch tensor. Padding: The sequence is padded to max_length=512, Truncation: The sequence is truncated if it exceeds max_length=512., Special Tokens: Special tokens are added by default.

for `input_testing1`: Return Type: Dictionary of lists (default behavior), Padding: The sequence is not padded to a fixed length, Truncation: The sequence is truncated if it exceeds max_length1, Special Tokens: Special tokens are not added.


# Model

- Transformer package: https://huggingface.co/transformers/v3.0.2/training.html 
- ![](Pasted%20image%2020241218135156.png)![](Pasted%20image%2020241218135213.png)


## Fvcore

- Fvcore.common.param_scheduler: https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#module-fvcore.common.checkpoint ![](Pasted%20image%2020241218151619.png)![](Pasted%20image%2020241218151630.png)


## gemma architecture official code

- https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L164
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
- https://github.com/google-deepmind/gemma/tree/main/gemma
- transformer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py

## Gemma2 architecture related ques 

- [model support] please support gemma2 https://github.com/NVIDIA/TensorRT-LLM/issues/1984
- could not finetune gemma 2 9b with lora and fsdp: https://github.com/huggingface/peft/issues/2111
- 


## Gemma-cookbook

- https://github.com/google-gemini/gemma-cookbook/tree/main
- About: A collection of guides and examples for the Gemma open models from Google.

## llama architecture
- https://github.com/meta-llama/llama3/blob/main/llama/model.py
- https://github.com/arpytanshu/tests/blob/b536ef106130707fb1b337a2f53be87a64a2fd0e/customForward/modeling_llama.py#L194


## transformer Engine

- https://github.com/NVIDIA/TransformerEngine
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html
- Accelerating a Hugging Face Llama 2 model with Transformer Engine: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-1.5/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html#:~:text=of%20almost%2040%25!-,Conclusion,up%20is%20even%20more%20pronounced!

# Low Precision Training with Kahan Summation
- [link](https://optimi.benjaminwarner.dev/kahan_summation/) 
- ![alt text](image.png)

# Accelerate training : **modeling_gemma2_fast.py** 

### Flash Attention
 [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention | PyTorch](https://pytorch.org/blog/flexattention/)

- https://github.com/Dao-AILab/flash-attention
- reading : https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
 
- `logit_softcapping`

### transformer engine 

- Improving Hugging Face Training Efficiency Through Packing with Flash Attention [link](https://huggingface.co/blog/packing-with-FA2)
- https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py#L447
# Faster inference: **modeling_gemma2.py** 

### Flash Attention
 [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention | PyTorch](https://pytorch.org/blog/flexattention/)

- https://github.com/Dao-AILab/flash-attention
- https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py#L1111
- https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py#L1160
- 
 
- `logit_softcapping` present in gemma2

### xtransformer

- toolbox to accelerate research on transformer - [link](https://github.com/facebookresearch/xformers/tree/main)
- 
- 

### using triton language kernels
- https://github.com/triton-lang/triton
- Lecture 14: Practitioners Guide to Triton [link](https://www.youtube.com/watch?v=DdTsX6DQk24)
- Lecture 29: Triton Internals [link](https://www.youtube.com/watch?v=njgow_zaJMw)
- Github Workshop | Triton NeuroTech [link](https://youtu.be/p5a7HgjeODw?si=q4JHPycwRmKnMLYV)



#### fused softmax
- ![](Pasted%20image%2020241221004839.png)
- https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py


## (Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)

- https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
- https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/fc133e4ffc6275f9d1c3a74ddd10e0a2/scaled_dot_product_attention_tutorial.ipynb#scrollTo=oUjAmpChvFXi

### # [Is pytorch SDPA using flash attention V2?](https://discuss.pytorch.org/t/is-pytorch-sdpa-using-flash-attention-v2/192165)

### FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
- https://tridao.me/blog/2024/flash3/



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

### gemma from lmdeploy
- https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/models/gemma.py

### blog to read
- supported model list:s https://lmdeploy.readthedocs.io/en/v0.2.5/supported_models/supported_models.html

### rms_norm
- https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/kernels/cuda/rms_norm.py

### kernels
- ![](Pasted%20image%2020241221010205.png)
  - cute kernel repo :
    - https://github.com/mayank31398/cute-kernels/tree/main/cute_kernels/kernels

## `xformers` -Toolbox to Accelerate Research on Transformers


# accelration using GPUs

### NVIDIA A100 Tensor Core GPU

- Unprecedented acceleration at every scale
- link - https://www.nvidia.com/en-in/data-center/a100/


# New reads to get the other flows of finetuning:
- https://github.com/georgian-io/LLM-Finetuning-Toolkit
- https://github.com/hiyouga/LLaMA-Factory
- https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb
- https://github.com/ashishpatel26/LLM-Finetuning
