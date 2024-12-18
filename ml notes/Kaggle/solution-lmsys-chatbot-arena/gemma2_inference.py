# %%writefile predict_m0.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
from human_pref.inference.modeling_gemma2 import Gemma2ForSequenceClassification
from human_pref.data.processors import ProcessorPAB
from human_pref.data.dataset import LMSYSDataset
from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator
from human_pref.utils import to_device


model_name_or_path = "/kaggle/input/lmsys-checkpoints-0-0805"
csv_path = "test.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
processor = ProcessorPAB( tokenizer=tokenizer, 
                         max_length=4096, 
                         support_system_role=False,)

dataset = LMSYSDataset(    
                       csv_file=csv_path,    
                       query=None,    
                       processor=processor,    
                       include_swap=False,    
                       is_parquet=True,)

dataloader = DataLoader(
    dataset,
    batch_size=80,
    num_workers=4,
    collate_fn=ShardedMaxTokensCollator(
        max_tokens=8192, base_collator=VarlenCollator()
    ),
)

# model for pipelined inference
num_hidden_layers = 42
device_map = {
    "model.embed_tokens": "cuda:0",
    "model.norm": "cuda:1",
    "score": "cuda:1",
}
for i in range(num_hidden_layers // 2):
    device_map[f"model.layers.{i}"] = "cuda:0"
for i in range(num_hidden_layers // 2, num_hidden_layers):
    device_map[f"model.layers.{i}"] = "cuda:1"

model = Gemma2ForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# inv_freq clones for each device
config = model.config
dim = config.head_dim
inv_freq = 1.0 / (
    config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
)
inv_freq0 = inv_freq.to("cuda:0")
inv_freq1 = inv_freq.to("cuda:1")


# for name, p in model.named_parameters():
#     print(name, p.device)
# for name, b in model.model.named_buffers():
#     print(name, b.device)

# pipeline parallelism with two GPUs
is_first = True
hidden_states = None
outs = []
for batch in tqdm(dataloader):
    for micro_batch in batch:
        input_ids = to_device(micro_batch["input_ids"], "cuda:0")
        seq_info = dict(
            cu_seqlens=micro_batch["cu_seqlens"],
            position_ids=micro_batch["position_ids"],
            max_seq_len=micro_batch["max_seq_len"],
            # attn_bias=BlockDiagonalCausalMask.from_seqlens(micro_batch["seq_lens"]),
        )
        seq_info = to_device(seq_info, "cuda:0")
        if is_first:
            with torch.no_grad(), torch.cuda.amp.autocast():
                prev_hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)
            is_first = False
            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, prev_hidden_states], "cuda:1"
            )
            continue
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
            hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)

            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, hidden_states], "cuda:1"
            )
            outs.append(logits.cpu())

# last micro-batch
with torch.no_grad(), torch.cuda.amp.autocast():
    logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
    outs.append(logits.cpu())

pred = torch.cat(outs, dim=0)
prob = pred.softmax(-1)
print(dataset.evaluate(prob.numpy()))

np.save('prob_m0.npy', prob)