import os
import torch
import json
from tensordict import TensorDict
from torch.utils.data import Dataset
from tensordict.persistent import PersistentTensorDict
from torch.utils.data import DataLoader

class NaiveTensorStore(Dataset):
    def __init__(self, 
                 memmap_name="disk_memmap", 
                 meta_name="meta.json"):
        """
        Torch tensor storage to map large data to disk.
        By default, Label=0 means it is only a placeholder and do not contain any real data.
        This class requires pre-defined tensor shape.
        """
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
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
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
        #assert idx < len(self.data)
        if idx >= self.save_pointer:
            raise IndexError
        return self.data[idx]
    
    def __len__(self):
        return self.save_pointer
    
    def append(self, item: dict):
        self.data[self.save_pointer] = item
        self.save_pointer += 1
        
    def set_item(self, idx, item: dict):
        assert idx < len(self.data)
        assert idx < self.save_pointer
        self.data[idx] = item
    
    def clear_storage(self):
        self.save_pointer = 0
    
    def save_to_disk(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        meta_path = os.path.join(save_dir, self.meta_name)
        with open(meta_path, 'w') as f:
            json.dump({
                'allocated_size': self.allocated_size,
                'config': self.config,
                'memmap_name': str(self.memmap_name),
                'meta_name': str(self.meta_name),
                "save_pointer": self.save_pointer
            }, f)
        self.data.memmap_()
    
    @classmethod
    def load_from_disk(cls, save_dir, meta_name="meta.json"):
        with open(os.path.join(save_dir, meta_name), 'r') as f:
            meta_info = json.load(f)
        storage = cls()
        storage.allocated_size = meta_info['allocated_size']
        storage.config = meta_info['config']
        storage.memmap_name = meta_info['memmap_name']
        storage.save_pointer = meta_info['save_pointer']
        storage.save_dir = save_dir
        #self.meta_name = meta_info['meta_name']
        
        storage.data = TensorDict.load(os.path.join(save_dir, storage.memmap_name))
        return storage
    
    def __del__(self):
        if self.save_dir is not None:
            self.save_to_disk()
    
    def get_data_loader(self, batch_size, feature_name, shuffle, num_workers, prefetch_factor=2):
        """
        This function is not class-specific but task-specific.
        But it is put here for interface convenience. 
        """
        def get_collate_fn(feature_name):
            def collate_fn(batch):
                feature = []
                labels = []
                for x in batch:
                    item = x.to_dict()
                    feature.append(item[feature_name])
                    labels.append(item['labels'])
                return_x = torch.stack(feature)
                labels = torch.stack(labels)
                return (return_x, labels)
            return collate_fn
        data_loader =  DataLoader(self, 
                                  batch_size=batch_size, 
                                  collate_fn=get_collate_fn(feature_name), 
                                  shuffle=shuffle, 
                                  num_workers=num_workers, 
                                  prefetch_factor=prefetch_factor)
        return data_loader
        

class VariedKeyTensorStore(Dataset):
    def __init__(self,
                 shelve_name="disk_shelve", 
                 meta_name="meta.json"):
        """
        Torch tensor storage to map large data to disk.
        This class works just like dict, but with similar interface to List.
        The underlying storage is PersistentTensorDict.
        """
        self.save_dir = None
        self.data = None
        self.shelve_name = shelve_name
        self.meta_name = meta_name
        self.save_pointer = 0
        self.index = []

    @classmethod
    def load_from_disk(cls, save_dir, meta_name="meta.json", open_mode="a"):
        with open(os.path.join(save_dir, meta_name), 'r') as f:
            meta_info = json.load(f)
        storage = cls()
        #storage.save_pointer = meta_info['save_pointer']
        storage.save_dir = save_dir
        storage.index = sorted(meta_info['index'], key=lambda x: int(x))
        storage.data = PersistentTensorDict(filename=os.path.join(save_dir, storage.shelve_name), mode=open_mode)
        return storage
    
    def save_to_disk(self, save_dir=None):
        if self.data is not None:
            if save_dir is None:
                save_dir = self.save_dir
            meta_path = os.path.join(save_dir, self.meta_name)
            with open(meta_path, 'w') as f:
                json.dump({
                    'meta_name': str(self.meta_name),
                    "save_pointer": self.save_pointer,
                    "save_dir": str(self.save_dir),
                    "shelve_name": str(self.shelve_name),
                    "index": self.index
                }, f)
        
    def init(self, save_dir):
        self.save_dir = save_dir
        save_path = os.path.join(save_dir, self.shelve_name)
        #self.data = shelve.open(save_path)
        self.data = PersistentTensorDict(filename=save_path, mode="w")
    
    def __len__(self):
        return len(self.index)

    def append(self, item: dict):
        self.data[str(self.save_pointer)] = item
        self.index.append(str(self.save_pointer))
        self.save_pointer += 1

    def set_item(self, idx, item: dict, overwrite=False):
        # If do not overwrite
        # The storage will leave the original data as it is 
        if isinstance(idx, int):
            assert str(idx) in self.index
            real_idx = self.index[idx]
            self.data[self.index[idx]] = item
        else:
            real_idx = idx
        if overwrite:
            self.data.del_(real_idx)
        self.data[real_idx] = item
    
    def keys(self):
        return self.index
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[self.index[idx]]
        else:
            return self.data[idx]    
    
    def __del__(self):
        self.save_to_disk()
        self.close()
    
    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None
    
    def open(self, path=None):
        if path is None:
            path = os.path.join(self.save_dir, self.shelve_name)
        self.data = PersistentTensorDict(filename=path, mode="a")

    def get_data_loader(self, batch_size, feature_name, shuffle, num_workers, prefetch_factor=2):
        """
        This function is not class-specific but task-specific.
        But it is put here for interface convenience. 
        """
        def collate_fn(batch_info):
            """
            Collate multiple data points into one batch.
            Return:
                original: Collected states of original code. 
                    Shape: (N, hidden_dim), where N is the sum of collected states in the given batch.
                mutated: Collected states of mutated code.
                    Shape: (M, hidden_dim), where M is the sum of collected states in the given batch.
                original_index: the index used in original for contrastive learning
                    shape: (Num), where Num is the number of instances in the given batch.
                mutated_index: the index used in mutated code for contrastive learning
                    shape: (Num), where Num is the number of instances in the given batch.
                ori_div_list: last token index of every original code instance in the batch 
            """
            original = list()
            mutated = list()
            original_index = []
            mutated_index = []
            original_shift_pointer = 0
            ori_div_list = []
            mutated_shift_pointer = 0
            for item in batch_info:
                item = item.to_dict()
                original.append(item["original"]["hidden_states"])
                original_length = item["original"]["hidden_states"].shape[0]
                for key in item["mutated_code"]:
                    original_index.append(original_shift_pointer + item["mutated_code"][key]["contrastive_list_original"])
                    mutated_index.append(mutated_shift_pointer + item["mutated_code"][key]["contrastive_list_mutated"])
                    mutated.append(item["mutated_code"][key]["hidden_states"])
                    mutated_shift_pointer += item["mutated_code"][key]["hidden_states"].shape[0]
                original_shift_pointer += original_length
                ori_div_list.append(original_shift_pointer - 1)
            return torch.cat(original), torch.cat(mutated), torch.cat(original_index), torch.cat(mutated_index), torch.tensor(ori_div_list)
        data_loader =  DataLoader(self, 
                                  batch_size=batch_size, 
                                  collate_fn=collate_fn, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers, 
                                  prefetch_factor=prefetch_factor)
        return data_loader