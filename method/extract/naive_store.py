import os
import torch
import json
from tensordict import TensorDict
from torch.utils.data import Dataset

class NaiveTensorStore(Dataset):
    def __init__(self, 
                 memmap_name="disk_memmap", 
                 meta_name="meta.json"):
        self.allocated_size = None
        self.config = None
        self.save_dir = None
        self.data = None
        self.memmap_name = memmap_name
        self.meta_name = meta_name
        self.save_pointer = 0
        
    
    def init(self, allocated_size, config, save_dir):
        self.allocated_size = allocated_size
        self.config = config
        self.save_dir = save_dir
        self.data = TensorDict(
            self.turn_config_into_dataset_key_value(config),
            batch_size=[]
        )
        self.data = self.data.expand(allocated_size)
        save_path = os.path.join(save_dir, self.memmap_name)
        self.data = self.data.memmap_like(save_path)
    
    def turn_config_into_dataset_key_value(self, config):
        dataset_key_value = {}
        for key, value in config.items():
            shape = value["shape"]
            dtype = getattr(torch, value["dtype"])
            dataset_key_value[key] = torch.zeros(shape, dtype=dtype)
        return dataset_key_value
    
    def __getitem__(self, idx):
        # While this upper bound might not be the actual length
        # We do not want to set additional safety checks
        assert idx < len(self.data)
        return self.data[idx]
    
    def __len__(self):
        return self.save_pointer
    
    def append(self, item: dict):
        self.data[self.save_pointer] = item
        self.save_pointer += 1
        
    def set_item(self, idx, item: dict):
        assert idx < len(self.data)
        self.data[idx] = item
        # While this might not be the actual length
        # We assume that the user knows what they are doing
        if idx >= self.save_pointer:
            self.save_pointer = idx + 1
    
    def clear_storage(self):
        self.save_pointer = 0
    
    def save_to_disk(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        meta_name = os.path.join(save_dir, self.meta_name)
        with open(meta_name, 'w') as f:
            json.dump({
                'allocated_size': self.allocated_size,
                'config': self.config,
                'memmap_name': self.memmap_name,
                'meta_name': self.meta_name,
                "save_pointer": self.save_pointer
            }, f)
        
    def load_from_disk(self, save_dir):
        with open(os.path.join(save_dir, self.meta_name), 'r') as f:
            meta_info = json.load(f)
        self.allocated_size = meta_info['allocated_size']
        self.config = meta_info['config']
        self.memmap_name = meta_info['memmap_name']
        self.save_pointer = meta_info['save_pointer']
        self.save_dir = save_dir
        #self.meta_name = meta_info['meta_name']
        
        self.data = TensorDict.load_memmap(os.path.join(self.save_dir, self.memmap_name))
    
        

