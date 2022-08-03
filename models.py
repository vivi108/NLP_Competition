
import torch
from torch import nn
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from utils import MODEL_CLASSES, MODEL_PATH_MAP
model_name = 'bert'
config_class, model_class, _ = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
#model_path = MODEL_PATH_MAP[model_name]
#modelClass_config = config_class.from_pretrained(model_path)


#TODO :batch만 input으로 들어온다
class COLA_model_baseline(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super().__init__()
        self.model_PLM = AutoModel.from_pretrained(pretrain_model_path) #[10,100]*3 -> [10,100,768]
        self.relu = nn.ReLU() 
        self.linear = nn.Linear(768,2)

    def forward(self, x):
        output= self.model_PLM(input_ids=x[0],token_type_ids=x[1], attention_mask=x[2])
        print(output[0].size(),output[1].size())
        output=output['last_hidden_state'][:, 0, :] 
        output=self.relu(output)
       
        output=self.linear(output) 
        print('Model output shape: ',output.shape)
        return output


class WiC_model_baseline(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super().__init__()
        self.model_PLM = model_class.from_pretrained(pretrain_model_path)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(768,2)

    def forward(self, x):
        output= self.model_PLM(input_ids=x[0],token_type_ids=x[1], attention_mask=x[2])
        print(output[0].size(),output[1].size())
        output=output['last_hidden_state'][:, 0, :] 
        output=self.sigmoid(output)
       
        output=self.linear(output) 
        print('Model output shape: ',output.shape)
        return output


class COPA_model_baseline(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super().__init__()
        self.model_PLM = model_class.from_pretrained(pretrain_model_path)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(768,1)
    def forward(self, x):
        output1= self.model_PLM(input_ids=x[0],token_type_ids=x[1], attention_mask=x[2])
        output2= self.model_PLM(input_ids=x[3],token_type_ids=x[4], attention_mask=x[5])
        # TODO : model_PLM을 거친 것들은 텐서가 아닌지? 왜 프린트가 안되죠..
        # print( output1.size())
        # print(output2.size())
        # output = torch.cat([output1, output2],  0)
        
        output1=output1['last_hidden_state'][:, 0, :] 
        output1=self.sigmoid(output1)
        output1=self.linear(output1) 
        output2=output2['last_hidden_state'][:, 0, :] 
        output2=self.sigmoid(output2)
        output2=self.linear(output2)
        # tuple =(output1, output2)
        print('Model output1 shape: ',output1.shape)
        print('Model output2 shape: ',output2.shape)
        output = torch.cat([output1, output2], dim=1)
        print('Model output shape: ',output.shape)
        
        return output


class BoolQ_model_baseline(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super().__init__()
        self.model_PLM = AutoModel.from_pretrained(pretrain_model_path)
        self.relu = nn.ReLU() 
        self.linear = nn.Linear(768,2)

    def forward(self, x):
        output= self.model_PLM(input_ids=x[0],token_type_ids=x[1], attention_mask=x[2])
        # print(output[0].size(),output[1].size())
        output=output['last_hidden_state'][:, 0, :] 
        output=self.relu(output)
        output=self.linear(output) 
        print('Model output shape: ',output.shape)
        return output


MODEL_DICT = {
    "cola_baseline": COLA_model_baseline,
    "wic_baseline": WiC_model_baseline,
    "copa_baseline": COPA_model_baseline,
    "boolq_baseline": BoolQ_model_baseline,
}