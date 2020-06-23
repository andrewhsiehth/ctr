import torch 
from torch.utils.data import Dataset 

from tqdm.auto import tqdm 

from collections import Counter 
import itertools 
import functools
import os 
import math 
import multiprocessing as mp 


class Criteo(Dataset):  
    NUM_FIELDS = 39 
    FIELDS_I = list(range(13)) 
    FIELDS_C = list(range(13, 39)) 

    CACHE_SAMPLE_OFFSETS = 'sample_offsets.cache.pt' 
    CACHE_FEATURE_MAPPING = 'feature_mapping.cache.pt'

    def __init__(self, data_path: str, min_threshold: int=10): 
        self.data_path = data_path 
        self.min_threshold = min_threshold 
        self.sample_offsets = None 
        self.feature_mapping = None 
        self.feature_default = None 
        
        self._locate_sample_offsets()  
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
        if os.path.exists(Criteo.CACHE_FEATURE_MAPPING):
            self.feature_mapping = torch.load(Criteo.CACHE_FEATURE_MAPPING) 
        else: 
            self.feature_mapping = [
                {
                    feature: feature_id for feature_id, feature in enumerate(
                        feature for feature, count in sorted(field.items()) if count >= self.min_threshold 
                    )
                } for field in self._count_field_features() 
            ] 
            torch.save(self.feature_mapping, Criteo.CACHE_FEATURE_MAPPING)
        self.feature_default = [len(m) for m in self.feature_mapping] 

    def _make_sample(self, line: str): 
        label, *values = line.rstrip(b'\n').split(b'\t') 
        for field_id in Criteo.FIELDS_I: 
            values[field_id] = self.feature_mapping[field_id].get(Criteo._quantize_I_feature(values[field_id]), self.feature_default[field_id]) 
        for field_id in Criteo.FIELDS_C: 
            values[field_id] = self.feature_mapping[field_id].get(values[field_id], self.feature_default[field_id]) 
        return torch.as_tensor(values), int(label)

    def _locate_sample_offsets(self, n_jobs: int=os.cpu_count()): 
        if os.path.exists(Criteo.CACHE_SAMPLE_OFFSETS): 
            self.sample_offsets = torch.load(Criteo.CACHE_SAMPLE_OFFSETS) 
            return 
        
        stat_result = os.stat(self.data_path) 
        chunk_size, _ = divmod(stat_result.st_size, n_jobs) 
        
        chunk_starts = [0]
        with open(self.data_path, mode='rb') as infile: 
            while infile.tell() < stat_result.st_size: 
                infile.seek(chunk_size, os.SEEK_CUR) 
                infile.readline() 
                chunk_starts.append(min(infile.tell(), stat_result.st_size)) 
        
        with mp.Pool(processes=n_jobs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool: 
            self.sample_offsets = list(itertools.chain.from_iterable(pool.imap(
                functools.partial(Criteo._locate_sample_offsets_job, self.data_path), 
                iterable=enumerate(zip(chunk_starts[:-1], chunk_starts[1:]))
            ))) 
        torch.save(self.sample_offsets, Criteo.CACHE_SAMPLE_OFFSETS)

    def _count_field_features(self, n_jobs: int=os.cpu_count()): 
        with mp.Pool(processes=n_jobs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool: 
            return list(map(
                functools.partial(functools.reduce, lambda x, y: x + y), 
                zip(*pool.imap_unordered(
                    functools.partial(Criteo._count_field_features_job, self.data_path), 
                    iterable=((i, self.sample_offsets[i::n_jobs]) for i in range(n_jobs))
                ))
            )) 

    @classmethod 
    def _locate_sample_offsets_job(cls, data_path: str, task: tuple): 
        job_id, (start, end) = task  
        offsets = [start] 
        with open(data_path, mode='rb') as infile: 
            infile.seek(start, os.SEEK_SET) 
            with tqdm(total=None, desc=f'[Loacate Sample Offsets] job: {job_id}', position=job_id) as pbar: 
                while infile.tell() < end: 
                    infile.readline() 
                    offsets.append(infile.tell()) 
                    pbar.update()  
            assert offsets.pop() == end 
        return offsets 

    @classmethod
    def _count_field_features_job(cls, data_path: str, task: tuple):
        job_id, sample_offsets = task  
        field_features_count = [Counter() for _ in range(Criteo.NUM_FIELDS)]
        with open(data_path, mode='rb') as infile:
            with tqdm(sample_offsets, desc=f'[Counting Field Features] job: {job_id}', position=job_id) as pbar: 
                for offset in pbar: 
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
    data_path='./data/criteo.dev.txt'
    dataset = Criteo(data_path) 
    print(dataset[0])
