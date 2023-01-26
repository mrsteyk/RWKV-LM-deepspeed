import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, args):
        self.args = args

        self.data = np.load(args.data_file).astype("int")
        
        # pad to the nearest multiple of ctx_len + 1
        d = args.ctx_len + 1
        rem = len(self.data) % d
        self.data = np.concatenate([self.data, np.zeros(d - rem)])
        self.data_split = np.split(self.data, len(self.data) // d)
        
        self.vocab_size = args.vocab_size
        self.data_size = len(self.data_split)
        #print("Data size:", self.data_size)
    
    # ???
    def __len__(self):
        # return self.data.count(0)
        return self.data_size
    
    def __getitem__(self, index):
        dix = self.data_split[index]
        if self.args.soft_emb_tune:
            return torch.tensor(np.concatenate(([-100]*self.args.soft_emb_tokens, dix[:-1])), dtype=torch.long), torch.tensor(np.concatenate(([-100]*self.args.soft_emb_tokens, dix[1:])), dtype=torch.long)
        return torch.tensor(dix[:-1], dtype=torch.long), torch.tensor(dix[1:], dtype=torch.long)
