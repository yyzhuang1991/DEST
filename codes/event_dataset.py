import pickle, torch, json, os, sys, re, numpy as np, copy, random
from os.path import abspath, dirname, join
from torch.utils.data import Dataset, DataLoader
from sys import stdout
from torch.autograd import Variable
from tabulate import tabulate 
from collections import Counter

from constants import label2ind, ind2label

class Event_Dataset(Dataset):
    def __init__(self, items, shuffle_seed = 100, shuffle = True):
        super(Event_Dataset, self).__init__()
        self.items = {}
        for name in items:
            self.items[name] = copy.deepcopy(items[name])
        if "labels" not in self.items:
            print("CAUTION: There is no given labels in the dataset. Making a fake event_labels now. ABORT if this is supposed to have true labels! ")
            self.items["labels"] = [-1] * len(self.items["events"])
        else:
            # make label integers
            self.items["labels"] = [label2ind[l] for l in self.items["labels"]]
        if shuffle:
            self.shuffle_seed = shuffle_seed
            self.shuffle_items()

    def shuffle_items(self):
        inds = list(range(len(self.items["events"])))
        random.Random(self.shuffle_seed).shuffle(inds)

        for name in self.items:
            if type(self.items[name]) != type([]):
                continue
            new_item = [self.items[name][idx] for idx in inds]
            self.items[name] = new_item        

    def __len__(self):
        return len(self.items["events"])

    def __getitem__(self, idx):
        d = {}
        for name in self.items:
            if type(self.items[name]) == type([]):
                d[name] = self.items[name][idx]
            else:
                d[name] = self.items[name]
        return d
        