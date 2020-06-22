import torch 
from torch.utils.data import Dataset 

from tqdm import tqdm 

from collections import Counter 
import functools
import os 
import math 
import multiprocessing as mp 


class Criteo(Dataset):  
    NUM_FIELDS = 39 
    FIELDS_I = list(range(13)) 
    FIELDS_C = list(range(13, 39)) 

    def __init__(self, data_path: str, min_threshold: int=10): 
        self.data_path = data_path 
        self.min_threshold = min_threshold 
        self.sample_offsets = self._locate_sample_offsets() 

        self._build_feature_mapping() 
        
    def __len__(self): 
        return len(self.sample_offsets) 

    def __getitem__(self, idx): 
        with open(self.data_path, mode='rb') as infile: 
            infile.seek(self.sample_offsets[idx]) 
            line = infile.readline() 
        return self._make_sample(line) 

    @property 
    def field_dims(self): 
        return [(f + 1) for f in self.feature_default]  
        
    def _build_feature_mapping(self): 
        self.feature_mapping = [
            {
                feature: feature_id for feature_id, feature in enumerate(
                    feature for feature, count in sorted(field.items()) if count >= self.min_threshold 
                )
            } for field in self._count_field_features() 
        ] 
        self.feature_default = [len(m) for m in self.feature_mapping] 

    def _make_sample(self, line: str): 
        label, *values = line.rstrip(b'\n').split(b'\t') 
        for field_id in Criteo.FIELDS_I: 
            values[field_id] = self.feature_mapping[field_id].get(Criteo._quantize_I_feature(values[field_id]), self.feature_default[field_id]) 
        for field_id in Criteo.FIELDS_C: 
            values[field_id] = self.feature_mapping[field_id].get(values[field_id], self.feature_default[field_id]) 
        return torch.as_tensor(values), int(label)

    def _locate_sample_offsets(self): 
        with open(self.data_path, mode='rb') as infile: 
            offsets = [
                infile.tell() for _ in tqdm(
                    iter(infile.readline, b''), 
                    desc='[Locate Line Offsets]'
                )
            ]
        offsets.pop(-1) 
        return [0] + offsets 

    def _count_field_features(self, n_processes: int=os.cpu_count()): 
        with mp.Pool(processes=n_processes) as pool: 
            return list(map(
                functools.partial(functools.reduce, lambda x, y: x + y), 
                zip(*tqdm(
                    pool.imap_unordered(
                        functools.partial(Criteo._process_job, self.data_path), 
                        iterable=(self.sample_offsets[i::n_processes] for i in range(n_processes))
                    ),  
                    desc='[Counting Field Features]'
                ))
            ))

    @classmethod
    def _process_job(cls, data_path: str, sample_offsets: list): 
        field_features_count = [Counter() for _ in range(Criteo.NUM_FIELDS)]
        with open(data_path, mode='rb') as infile:
            for offset in tqdm(sample_offsets, desc='[Process Job]'): 
                infile.seek(offset) 
                Criteo._count_one_line(infile.readline(), field_features_count) 
        return field_features_count 

    @classmethod 
    def _count_one_line(cls, line: str, field_features_count: list): 
        label, *values = line.rstrip(b'\n').split(b'\t') 
        for field_id in Criteo.FIELDS_I: 
                field_features_count[field_id][Criteo._quantize_I_feature(values[field_id])] += 1 
        for field_id in Criteo.FIELDS_C: 
                field_features_count[field_id][values[field_id]] += 1    
    
    @classmethod
    def _quantize_I_feature(cls, value: str): 
        if value == b'': 
            return 'NULL' 
        value = int(value) 
        if value > 2: 
            return str(int(math.log(value)**2)) 
        else: 
            return str(value - 2)

if __name__ == '__main__': 
    data_path='./criteo.dev.txt'
    dataset = Criteo(data_path)
    l1 = dataset._count_field_features(n_processes=4) 
    l2 = dataset._count_field_features(n_processes=1) 
    print(l1[0]) 
    print('=========')
    print(l2[0])
    print(l1 == l2)
    print(len(dataset))

    print(dataset[10])



