from calendar import EPOCH
from turtle import forward
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

from conifg import DATAPATH_DICT
from models import MODEL_DICT
from datasets import DATASET_DICT
from utils import set_seed, torch_MCC, torch_ACC

assert MODEL_DICT.keys() == DATASET_DICT.keys(), "Model and dataset not matched"

#set config and hyperparameter
task = "cola"
model_name = "cola_baseline" 
pretrain_model_path = "klue/bert-base" #korean bert https://huggingface.co/klue/bert-base
save_path = "result/tmp"
gpu = "cuda:0"
device = torch.device(gpu)
batch_size = 300
max_length = 40
lr = 2e-5
eps = 1e-8
EPOCH = 30

#set seed
seed = 42
set_seed(seed)

train_data_path = DATAPATH_DICT[task]["train"]
val_data_path = DATAPATH_DICT[task]["val"]

#load dataset and dataloader
train_dataset = DATASET_DICT[model_name](train_data_path, pretrain_model_path, max_length)
val_dataset = DATASET_DICT[model_name](val_data_path, pretrain_model_path, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

#load model
model = MODEL_DICT[model_name](pretrain_model_path)
model.to(device)

#set criterion
lf = CrossEntropyLoss()
lf.to(device)

#set optimizer
optimizer = Adam(model.parameters(), lr=lr, eps=eps)
#파이썬에서 폴더 또는 경로 생성
os.makedirs(save_path, exist_ok=True)

result_json = {"epoch":{}}

for e in range(EPOCH):
    print("\n\n---- epoch {} ----".format(e))
    
    #train_phase
    all_pred = []
    all_label = []
    model.train()
    for batch in tqdm(train_dataloader, desc="train phase"):
        optimizer.zero_grad()

        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
        
        pred = model(batch)
        
        #save all result
        all_pred.append(pred)
        #print(pred.shape)
        all_label.append(batch[-1])
        #print(batch[-1])

        loss = lf(pred, batch[-1])
        loss.backward()

        optimizer.step()

    #check performance
    all_pred = torch.argmax(torch.cat(all_pred, dim=0), dim=-1)
    all_label = torch.cat(all_label, dim=0)
    train_acc = torch_ACC(all_pred, all_label)
    train_mcc = torch_MCC(all_pred, all_label)

    print("train result - acc:{:.4f},\t mcc:{:.4f}".format(train_acc, train_mcc))

    #validation phase
    all_val_pred = []
    all_val_label = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="validation phase"):

            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
                # input_ids = i[0]
                # token_type_ids = i[1]
                # attention_mask = i[2]

            pred = model(batch)

            all_val_pred.append(pred)
            all_val_label.append(batch[-1])

    #check performance
    all_val_pred = torch.argmax(torch.cat(all_val_pred, dim=0), dim=-1)
    all_val_label = torch.cat(all_val_label, dim=0)
    val_acc = torch_ACC(all_val_pred, all_val_label)
    val_mcc = torch_MCC(all_val_pred, all_val_label)
    print("validation result - acc:{:.4f},\t mcc:{:.4f}".format(val_acc, val_mcc))

    #saving model
    result_json["epoch"][e] = {"train_acc":train_acc, "train_mcc":train_mcc, "val_acc":val_acc, "val_mcc":val_mcc}
    torch.save(model.state_dict(), os.path.join(save_path, '{}.model'.format(e)))
    print("saving model at {}...".format(os.path.join(save_path, '{}.model'.format(e))))
    print("--------------------")
    

json.dump(result_json, open(os.path.join(save_path, 'result.json'), "w"), indent=4)

#save setting
setting_json = {"task":task, "model_name":model_name, "pretrain_model_path":pretrain_model_path, "batch_size":batch_size, "max_len":max_length, "lr":lr, "eps":eps}
json.dump(setting_json, open(os.path.join(save_path, 'setting.json'), "w"), indent=4)
