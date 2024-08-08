from pathlib import Path
import os, sys
import random
import re

# pos_label设置为1，代表标签为1的样本是正例，标签为2的样本是负例。

import numpy as np
import pandas as pd
from tqdm import tqdm

from customDataset import CustomDataset
from utils import cal_best_PRF
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
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

ROOT_DIR = Path(__file__).parent
resPath = os.path.join(ROOT_DIR,'result.csv')

device = torch.device("cuda:0")

llm_path = "/home/lenovo/guanwei/SemAD/Meta-Llama-3-8B"

step = 'N'
peft_path = "/home/lenovo/guanwei/SemAD/llama3-8b-int4-dolly_"+ str(step)

# data_path = '/home/lenovo/guanwei/SemAD/test_data.csv'
data_path = '/home/lenovo/guanwei/SemAD/eval_data.csv'

max_sequence_len = 1024

batch_size = 8


tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="right")
# tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=False,  # use double quantition
    bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
)

# model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, device_map='auto')

# model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=bnb_config, use_cache=False,
#                                              device_map='auto')

model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=bnb_config, torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True, device_map = device)

if peft_path is not None:
    print(f'load {peft_path}')
    model = PeftModel.from_pretrained(
        model,
        peft_path,
        torch_dtype=torch.float16,
    )

########################################################
print(f'dataset: {data_path}')

test_dataset = CustomDataset(data_path)


trace_gt, event_gt_ = test_dataset.get_label()


event_gt = []
for element in event_gt_:
    event_gt.append(np.fromstring(element[1:-1], sep=' ',dtype=int))


trace_gt =np.array(trace_gt).astype(int)

model.eval()

start_time = time.time()

indexes = [i for i in range(len(test_dataset))]  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def custom_mean(data, alpha = 0.6):
    num = 0
    res = 0

    for i in range(len(data) - 1, -1, -1):
        res = alpha * res + data[i]
        num = alpha * num + 1

    return res / num

pre = 0
scores = []
event_level_score = []
with torch.no_grad():
    for bathc_i in tqdm(range(batch_size, len(indexes) + batch_size, batch_size)):
        if bathc_i <= len(indexes):
            this_batch_indexes = list(range(pre, bathc_i))
        else:
            this_batch_indexes = list(range(pre, len(indexes)))
        trace_batch, _  = test_dataset.get_batch(this_batch_indexes)
        
        
        
        
        # trace_batch = ['Log in, Select update player details, Select player, Update player information, Submit changes']
        
        llm_ids = tokenizer(list(trace_batch), return_tensors="pt", max_length=max_sequence_len, padding=True, truncation=True).input_ids
        target_llm_ids = torch.cat([llm_ids[:,1:], torch.full((len(this_batch_indexes),1), tokenizer.eos_token_id, device=llm_ids.device)], dim=-1).numpy()    # add eos token

        llm_ids=llm_ids.to(device)

        output = model(llm_ids)

        outputs = output.logits
        
        outputs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()

        mask = ~(llm_ids == tokenizer.eos_token_id).detach().cpu().numpy()


        for ith in range(len(this_batch_indexes)):
            scores.append([])
            event_level_score.append([])
            valid_llm_ids = target_llm_ids[ith][mask[ith]]
            valid_outputs = outputs[ith][mask[ith]]

            indices = np.where(valid_llm_ids == 1174)[0]  #1174 is 'Ġ,'
            indices = np.concatenate((indices,np.array([valid_llm_ids.shape[0]-1])))

            split_probs = np.split(valid_outputs, indices, axis=0)
            split_llm_ids = np.split(valid_llm_ids, indices)

            for i, sub in  enumerate(split_probs):
                if i>0 and i<len(split_probs)-1:
                    split_probs[i] = sub[1:]

            for i, sub in  enumerate(split_llm_ids):
                if i>0 and i<len(split_llm_ids)-1:
                    split_llm_ids[i] = sub[1:]

            for j in range(len(split_llm_ids)):
                p_distribution =  split_probs[j]
                p = split_probs[j][list(range(len(split_llm_ids[j]))),split_llm_ids[j]]
                temp_scores = np.sum(np.log(p_distribution) * p_distribution, 1) - np.log(p)
                temp_score = sigmoid(custom_mean(temp_scores))
                # if j == 0:
                    # temp_score = 0
                scores[-1].append(temp_score)

                if j < len(split_llm_ids)-1:
                    event_level_score[-1].append(temp_score)

        pre = bathc_i


trace_level_score = []
for score in scores:
    trace_level_score.append(np.array(score).max())
trace_level_score = np.array(trace_level_score)
event_level_score = np.concatenate(event_level_score)


end_time = time.time()

run_time=end_time-start_time
print('run_time')
print(run_time)

event_gt = np.concatenate(event_gt)

# threshold = 0.85
threshold = None

if threshold is None:
    trace_p, trace_r, trace_f1, trace_aupr, trace_threshold = cal_best_PRF(trace_gt, trace_level_score)


    print("Trace-level anomaly detection")
    print(f'precision: {trace_p}, recall: {trace_r}, F1-score: {trace_f1}, trace_threshold: {trace_threshold} ,AP: {trace_aupr}')


    event_p, event_r, event_f1, event_aupr, event_threshold = cal_best_PRF(event_gt, event_level_score)


    print("Event-level anomaly detection")
    print(f'precision: {event_p}, recall: {event_r}, F1-score: {event_f1}, event_threshold:{event_threshold} AP: {event_aupr}')

    datanew = pd.DataFrame([{'index':data_path,'trace_p': trace_p, "trace_r": trace_r,'trace_f1':trace_f1,'trace_threshold':trace_threshold,'trace_aupr':trace_aupr,
                                    'event_p': event_p, "event_r": event_r, 'event_f1': event_f1,'event_threshold':event_threshold, 'event_aupr': event_aupr, 'time':run_time
                                    }])
    if os.path.exists(resPath):
        data = pd.read_csv(resPath)
        data = pd.concat([data,datanew],ignore_index=True)
    else:
        data = datanew
    data.to_csv(resPath ,index=False)
else:
    all_ad_Pred = np.where(trace_level_score > threshold, 1, 0)
    event_level_preds = np.where(event_level_score > threshold, 1, 0)

    precision = precision_score(trace_gt, all_ad_Pred, average="binary", pos_label=1)
    recall = recall_score(trace_gt, all_ad_Pred, average="binary", pos_label=1)
    f = f1_score(trace_gt, all_ad_Pred, average="binary", pos_label=1)

    acc = accuracy_score(trace_gt, all_ad_Pred)

    num_anomalous = (trace_gt == 1).sum()
    num_normal = (trace_gt == 0).sum()

    print(f'Number of anomalous traces: {num_anomalous}; number of normal traces: {num_normal}')

    det_num_anomalous = (all_ad_Pred == 1).sum()
    det_num_normal = (all_ad_Pred == 0).sum()

    print(f'Number of detected anomalous traces: {det_num_anomalous}; number of detected normal traces: {det_num_normal}')

    print(f'precision: {precision}, recall: {recall}, f1: {f}, accuracy: {acc}')


    event_precision = precision_score(event_gt, event_level_preds, average="binary", pos_label=1)
    event_recall = recall_score(event_gt, event_level_preds, average="binary", pos_label=1)
    event_f = f1_score(event_gt, event_level_preds, average="binary", pos_label=1)

    event_acc = accuracy_score(event_gt, event_level_preds)

    print(f'event_precision: {event_precision}, recall: {event_recall}, f1: {event_f}, accuracy: {event_acc}')
