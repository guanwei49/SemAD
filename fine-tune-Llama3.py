import os, sys
import random

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from customDataset import CustomDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from torch import optim
import torch
import torch.nn as nn
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)

from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import DataLoader

def print_number_of_trainable_model_parameters(model):
    params = set()
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            params.add(param)
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return params


device = torch.device("cuda:0")

llm_path = "/home/lenovo/guanwei/SemAD/Meta-Llama-3-8B"

data_path='/home/lenovo/guanwei/SemAD/training_data.csv'

data_name = os.path.splitext(os.path.basename(data_path))[0]

ft_model_name = "/home/lenovo/guanwei/SemAD/llama3-8b-int4-dolly"


max_sequence_len = 512
min_sequence_len = 1

batch_size = 8
lr = 5e-5
epochs = 10

tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=True,  # use double quantition
    bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
)

# model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, device_map='auto')

model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=bnb_config, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device)


model = prepare_model_for_kbit_training(model)
'''
- r, the dim of the low_rank matrices
- lora_alpha, scaling factor, the weight is scaled by lora_alpha/r, 
  the higher value assigns more weight to the LoRA activations
- target_modules: default is "q_proj", "v_proj"
- bias, the recommend setting bias to None first, and then lora_only, before trying all.
'''
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

### compare trainable parameters
trainable_model_params = print_number_of_trainable_model_parameters(model)


# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(trainable_model_params, lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.8)

train_dataset = CustomDataset(data_path)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss(reduction='mean')

steps = 0
for epoch in range(epochs):
    indexes = [i for i in range(len(train_dataset))]  # 打乱顺序
    random.shuffle(indexes)

    total_acc, total_count, train_loss = 0, 0, 0

    for bathc_i in tqdm(range(batch_size, len(indexes) + 1, batch_size)):
        steps += 1
        this_batch_indexes = indexes[bathc_i - batch_size:bathc_i]
        trace_batch, _  = train_dataset.get_batch(this_batch_indexes)

        # trace_batch_ = []
        # for e in trace_batch:
        #     trace_batch_.append(e.replace(' , ',', '))
        
        # trace_batch = trace_batch_
        
        optimizer.zero_grad()
        llm_ids = tokenizer(list(trace_batch), return_tensors="pt", max_length=max_sequence_len, padding=True, truncation=True).input_ids.to(device)
        target_llm_ids = torch.cat([llm_ids[:,1:], torch.full((batch_size,1), tokenizer.eos_token_id, device=llm_ids.device)], dim=-1)    # add eos token
        
        # for j in range(len(trace_batch)):
        #     if  trace_batch[j].count(',') != (llm_ids[j]==11).sum().detach().cpu():
        #         print('jj')

        output = model(llm_ids)
            
        outputs = output.logits

        mask = ~(llm_ids == tokenizer.eos_token_id)

        loss = criterion(outputs[mask], target_llm_ids[mask])

        loss.backward()
        optimizer.step()

        total_acc += (outputs[mask].argmax(1) == target_llm_ids[mask]).sum().item()
        train_loss += loss.item()*target_llm_ids[mask].size(0)

        total_count +=  target_llm_ids[mask].size(0)

        if steps % 500 == 0 :
            model.save_pretrained(ft_model_name+'_'+str(steps))

    train_loss_epoch = train_loss / total_count
    train_acc_epoch = total_acc / total_count
    print(f"[Epoch {epoch + 1:{len(str(epochs))}}/{epochs}] "
                f"[loss: {train_loss_epoch:3f}]"
                f"[acc: {train_acc_epoch:3f}]")

    scheduler.step()

model.save_pretrained(ft_model_name+'_'+str(steps))