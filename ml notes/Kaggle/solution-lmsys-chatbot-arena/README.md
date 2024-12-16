# LMSYS - Chatbot Arena Human Preference Predictions

[Competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

## Requirements

### Hardware

A100 SXM 80G x4

### Software

Base Image
```
nvcr.io/nvidia/pytorch:24.04-py3
```

Packages
```
detectron2==0.6
transformers==4.43.3
datasets==2.19.0
flash-attn==2.6.2
optimi==0.2.1
```


## Training

Directory structure should be as follows.

```
├── data
│   ├── train.csv
│   └── test.csv
├── artifacts
│   ├── dtrainval.csv
│   ├── lmsys-33k-deduplicated.csv
│   ├── ...
│   ├── stage1
│   ├── ...
│   └── stage3
└── src  # this repo
    ├── configs
    ├── human_pref
    └── main.py
```

1.  `python scripts/prepare_dataset.py` and download 21k external data from  [abdullahmeda](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)
2. [stage1](configs/stage1/README.md)
3. [make pseudo labels](configs/stage1_generate_pseudo_labels/README.md)
4. [stage2](configs/stage2/README.md)
4. [stage3](configs/stage3/README.md)

## Inference

Reference scripts to convert checkpoints for inference.
```
python scripts/prepare_gemma2_for_submission.py
python scripts/prepare_llama3_for_submission.py
```


[Kaggle Notebook](https://www.kaggle.com/code/tascj0/lmsys-0805)
# Solution

Our [inference code](https://www.kaggle.com/code/tascj0/lmsys-0805/notebook) and [training code](https://github.com/tascj/kaggle-lmsys-chatbot-arena) are made public.
## Baseline

- We used `StratifiedGroupKFold` based on prompt, reserving 20% as a validation set.
- 21k data from the deduplicated 33k dataset were added to the training data, thanks to [@abdullahmeda](https://www.kaggle.com/abdullahmeda).
- Instructions from [2306.05685](https://arxiv.org/abs/2306.05685) was used to format the input. We tried both prompt-res_a-res_b and promt-res_a-prompt-res_b. For 1.5B models, the latter seemed better, while for models 7B and above, there wasn't much difference. Considering token efficiency, we mainly used the PAB format. Max sequence length for gemma2-9b is around 4340.
- A custom head is used for classification. `XXXForSequenceClassification` initialization of head gives high loss on early iterations, so the head is re-initialized after model initialized.

```
    model.score = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hdim, hdim // 2),
        torch.nn.Dropout(0.1),
        torch.nn.GELU(),
        torch.nn.Linear(hdim // 2, 3),
    )
```

Based on my experience on previous competitions, I did not try LoRA and used only full-parameter training. With BF16 and optimizer with [kahan summation](https://optimi.benjaminwarner.dev/kahan_summation/) support, it's possible to train 7B models using single A100 80G, 9B models requires two A100s.

During the last 10 days of this competition, I used A100 80G x4 for all experiments.

## Full Swap

In early experiments, I performed random swaps of response_a and response_b as augmentation, which improves val log_loss a bit. [@kapenon](https://www.kaggle.com/kapenon) found that including both the original sample and its swap was better. To avoid overfitting, the gradients of the original sample and its swap must be accumulated for the same `optimizer.step`. Although the training time doubles, full swap shows a stable 0.003 improvement for gemma2-9b compared to random swap.  
Furthermore, I tried adding different input formats (PAB and PAPB) as augmentation in the same way, bringing a small (0.001) improvement.

## Steps to Train the Final Model

### Stage 1

We fine-tuned `google/gemma-2-9b-it`, `google/gemma-2-27b-it`, and `RLHFlow/ArmoRM-Llama3-8B-v0.1`. The validation log losses were 0.891, 0.883, and 0.899 respectively without TTA. After average ensemble, the log loss was 0.876.

After completing the `gemma-2-9b` training, I spent a lot of time trying `gemma-2-27b` without getting good results. By comparing with [@kapenon](https://www.kaggle.com/kapenon)'s code, I adjusted the batch_size to 80 and turned off grad_clip, finally successfully trained the model.

### Stage 2 (pseudo labeling)

We generated pseudo labels for 240k data using the ensemble obtained from stage 1. Of this, 110k data came from lmsys-1m (prepared by [@kapenon](https://www.kaggle.com/kapenon)), 130k came from other datasets (@liushuzhi tested numerous external datasets using a 1.5B model).  
On this dataset, we fine-tuned `gemma-2-9b` and `RLHFlow/ArmoRM-Llama3-8B-v0.1`.  
From this stage, I turned off the window attention for `gemma-2-9b`, because I'm not sure if efficient attention implementations supporting sm75 could do window attention. The longest input length was 4340 (including instruction), so this should have minimal impact on the score.

TTA was not applied when generating pseudo labels (my mistake).

### Stage 3

Based on the checkpoint obtained in stage 2, we fine-tuned using 55k+21k data. Multi input format augmentation was disabled due to time limit.  
On the 20% validation set, the two models achieved 0.884 and 0.890 respectively, with the average ensemble log loss being 0.876~0.877.  
At submission, input for `RLHFlow/ArmoRM-Llama3-8B-v0.1` was AB swapped. This model scored 0.873 on the old LB. After training with all data, this model reached 0.869 on the old LB. After adjusting the ensemble ratio to 2:1, the score was 0.868.

# Faster training

We used `flash-attn==2.6.2` for its `logit_softcapping` support.

When using `flash_attn_varlen_func`, attention_mask and padding are unnecessary. To avoid wasting computation on pad tokens, I

1. Implemented a custom collator to perform sequence concat on samples and prepare `cu_seqlens`
2. Modified the code based on huggingface's implementation, so the model only accepts `input_ids` and `cu_seqlens`. The model does not involve padding from start to finish.

Additionally, `RMSNorm` and `FusedRoPEFunc` from `transformer_engine` were used to further accelerate training.

# Faster inference

T4x2 is sufficient to run 7b-9b models in fp16. Transformers can be almost evenly distributed across 2 GPUs, just needing slight code modifications to make executions on two GPUs pipelined.

Using latest efficient operators on T4 (sm75) doesn't seem easy. After some attempts, I used the following triton operators for inference:

1. `context_attention_fwd` from `ModelTC/lightllm`, with some optimizations and `logit_softcapping` support.
2. `rms_norm` and `fused_rotary_emb` from `InternLM/lmdeploy`
3. `gelu_and_mul_fwd` and `silu_and_mul_fwd` from `ModelTC/lightllm`

`memory_efficient_attention` from `xformers` was used for Llama3.

Same as training, the entire inference process is also based on sequence collate, requiring no padding.


Kaggle post : [link](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)