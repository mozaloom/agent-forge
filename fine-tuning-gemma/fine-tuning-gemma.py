import os
from enum import Enum
from functools import partial
import pandas as pd
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig
from peft import LoraConfig, TaskType
from dotenv import load_dotenv
from utils import preprocess
from peft import LoraConfig
from trl.trainer.sft_trainer import SFTTrainer as _BaseSFTTrainer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

seed = 42
set_seed(seed)

model_name = "google/gemma-2-2b-it"
dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"


dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), remove_columns="messages")
dataset = dataset["train"].train_test_split(0.1)

# Modify the tokenizer to include special tokens for function calling and thinking
class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call="<tool_call>"
    eotool_call="</tool_call>"
    tool_response="<tool_reponse>"
    eotool_response="</tool_reponse>"
    pad_token = "<pad>"
    eos_token = "<eos>"
    @classmethod
    def list(cls):
        return [c.value for c in cls]

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list()
    )
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             attn_implementation='eager',
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16)  # Set dtype during loading
model.resize_token_embeddings(len(tokenizer))

#LoRA configuration
# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 16
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 64
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(r=rank_dimension,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"], # wich layer in the transformers do we target ?
                         task_type=TaskType.CAUSAL_LM)

# the Trainer and the Fine-Tuning hyperparameters
username="mozaloom"# REPLCAE with your Hugging Face username
output_dir = "gemma-2-2B-it-thinking-function_calling-V0" # The directory where the trained model checkpoints, logs, and other artifacts will be saved. It will also be the default name of the model when pushed to the hub if not redefined later.
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4
logging_steps = 5
learning_rate = 1e-4 # The initial learning rate for the optimizer.

max_grad_norm = 1.0
num_train_epochs=1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 1500

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    bf16=True,
    hub_private_repo=False,
    push_to_hub=False,
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=True,
    max_seq_length=max_seq_length,
)

class SFTTrainer(_BaseSFTTrainer):
    def _move_model_to_device(self, model, device):
        # Use to_empty() so meta tensors get real storage on `device`
        return model.to_empty(device=device)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Start the training process
trainer.train()
trainer.save_model()

trainer.push_to_hub(
    repo_id=f"{username}/{output_dir}",
    commit_message="Pushing trained model to Hugging Face Hub",
    private=True,
)

tokenizer.eos_token = "<eos>"
# push the tokenizer to hub 
tokenizer.push_to_hub(
    f"{username}/{output_dir}",
    commit_message="Pushing tokenizer to Hugging Face Hub",
    private=True
)









