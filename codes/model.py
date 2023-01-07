from transformers import AutoModel, AutoConfig
import torch 
import sys
from os.path import join, dirname, abspath
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(torch.nn.Module):
    def __init__(self, model_type, dropout = 0): 
        super(Model, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.config = AutoConfig.from_pretrained(model_type)
        self.encoder_dimension = self.config.hidden_size
        self.soft = torch.nn.Softmax(dim=1)    
        self.drop = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(self.encoder_dimension, 3)

    def forward(self, item): 

        _, event_embds = self.get_event_embds(item)

        logits = self.linear(event_embds)
        logits = self.drop(logits)
        return [logits]

    def get_event_embds(self, item):
        input_ids = item["event_ids"].to(device)
        attention_mask = item["event_masks"].to(device) if 'event_masks' in item else None
        last_hidden_states, event_embds = self.encoder(input_ids, attention_mask = attention_mask,return_dict = False)
        return last_hidden_states, event_embds

    
   