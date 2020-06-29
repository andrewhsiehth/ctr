import torch 
from torch.utils.data import Dataset 

import numpy as np

from tqdm.auto import tqdm 

import typing 
from typing import List
from typing import Dict 
from typing import Tuple 

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

    TRAIN_FILE_NAME = 'train.txt' 
    TEST_FILE_NAME = 'test.txt' 

    CACHE_SAMPLE_OFFSETS_TRAIN = 'sample_offsets_train.cache.pt' 
    CACHE_SAMPLE_OFFSETS_TEST = 'sample_offsets_test.cache.pt' 
    CACHE_FEATURE_MAPPING = 'feature_mapping.cache.pt'

    def __init__(self, data_path: str, sample_offsets: np.ndarray, feature_mapping: List[Dict[bytes, int]], feature_default: List[Dict[bytes, int]]): 
        self.data_path = data_path   
        self.sample_offsets = sample_offsets  
        self.feature_mapping = feature_mapping  
        self.feature_default = feature_default  
        
    def __len__(self) -> int: 
        return len(self.sample_offsets) 

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: 
        with open(self.data_path, mode='rb') as infile: 
            infile.seek(self.sample_offsets[idx]) 
            line = infile.readline() 
        return self._make_sample(line) 

    @property 
    def field_dims(self) -> List[int]: 
        return [(f + 1) for f in self.feature_default]  
    
    def _make_sample(self, line: bytes) -> Tuple[torch.Tensor, torch.Tensor]: 
        label, *values = line.rstrip(b'\n').split(b'\t') 
        for field_id in Criteo.FIELDS_I: 
            values[field_id] = self.feature_mapping[field_id].get(Criteo._quantize_I_feature(values[field_id]), self.feature_default[field_id]) 
        for field_id in Criteo.FIELDS_C: 
            values[field_id] = self.feature_mapping[field_id].get(values[field_id], self.feature_default[field_id]) 
        return torch.as_tensor(values), torch.as_tensor([float(label)]) 

    @classmethod 
    def prepare_Criteo(cls, root: str, min_threshold: int=10, n_jobs: int=os.cpu_count()) -> Tuple[Dataset, Dataset]: 
        if os.path.exists(Criteo.CACHE_SAMPLE_OFFSETS_TRAIN): 
            sample_offsets_train = torch.load(Criteo.CACHE_SAMPLE_OFFSETS_TRAIN)
        else: 
            sample_offsets_train = Criteo._locate_sample_offsets(
                data_path=os.path.join(root, Criteo.TRAIN_FILE_NAME), 
                n_jobs=n_jobs
            ) 
            torch.save(sample_offsets_train, Criteo.CACHE_SAMPLE_OFFSETS_TRAIN)
        
        if os.path.exists(Criteo.CACHE_FEATURE_MAPPING): 
            feature_mapping = torch.load(Criteo.CACHE_FEATURE_MAPPING) 
        else: 
            feature_mapping = Criteo._build_feature_mapping(
                data_path=os.path.join(root, Criteo.TRAIN_FILE_NAME), 
                sample_offsets=sample_offsets_train, 
                min_threshold=min_threshold,
                n_jobs=n_jobs 
            ) 
            torch.save(feature_mapping, Criteo.CACHE_FEATURE_MAPPING) 
        feature_default = Criteo._build_feature_default(feature_mapping=feature_mapping) 

        if os.path.exists(Criteo.CACHE_SAMPLE_OFFSETS_TEST): 
            sample_offsets_test = torch.load(Criteo.CACHE_SAMPLE_OFFSETS_TEST) 
        else: 
            sample_offsets_test = Criteo._locate_sample_offsets(
                data_path=os.path.join(root, Criteo.TEST_FILE_NAME), 
                n_jobs=n_jobs 
            )
            torch.save(sample_offsets_test, Criteo.CACHE_SAMPLE_OFFSETS_TEST) 
        
        dataset_train = Criteo(
            data_path=os.path.join(root, Criteo.TRAIN_FILE_NAME), 
            sample_offsets=sample_offsets_train, 
            feature_mapping=feature_mapping, 
            feature_default=feature_default 
        ) 
        dataset_test = Criteo(
            data_path=os.path.join(root, Criteo.TEST_FILE_NAME), 
            sample_offsets=sample_offsets_test, 
            feature_mapping=feature_mapping, 
            feature_default=feature_default 
        )
        return dataset_train, dataset_test 

    @classmethod 
    def _build_feature_mapping(cls, data_path: str, sample_offsets: np.ndarray, min_threshold: int, n_jobs: int) -> List[Dict[bytes, int]]: 
        return [
            {
                feature: feature_id for feature_id, feature in enumerate(
                    feature for feature, count in sorted(field.items()) if count >= min_threshold 
                )
            } for field in Criteo._count_field_features(data_path=data_path, sample_offsets=sample_offsets, n_jobs=n_jobs) 
        ] 

    @classmethod
    def _build_feature_default(cls, feature_mapping: List[Dict[bytes, int]]) -> List[int]: 
        return [len(m) for m in feature_mapping] 

    @classmethod 
    def _locate_sample_offsets(cls, data_path: str, n_jobs: int) -> np.ndarray: 
        stat_result = os.stat(data_path) 
        chunk_size, _ = divmod(stat_result.st_size, n_jobs) 
        
        chunk_starts = [0]
        with open(data_path, mode='rb') as infile: 
            while infile.tell() < stat_result.st_size: 
                infile.seek(chunk_size, os.SEEK_CUR) 
                infile.readline() 
                chunk_starts.append(min(infile.tell(), stat_result.st_size)) 
        chunk_starts = np.array(chunk_starts)

        with mp.Pool(processes=n_jobs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool: 
            return np.array(list(itertools.chain.from_iterable(pool.imap(
                functools.partial(Criteo._locate_sample_offsets_job, data_path), 
                iterable=enumerate(zip(chunk_starts[:-1], chunk_starts[1:]))
            )))) 

    @classmethod 
    def _count_field_features(cls, data_path: str, sample_offsets: np.ndarray, n_jobs: int) -> List[typing.Counter[bytes]]: 
        with mp.Pool(processes=n_jobs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool: 
            return list(map(
                functools.partial(functools.reduce, lambda x, y: x + y), 
                zip(*pool.imap_unordered(
                    functools.partial(Criteo._count_field_features_job, data_path), 
                    iterable=((i, sample_offsets[i::n_jobs]) for i in range(n_jobs))
                ))
            )) 

    @classmethod 
    def _locate_sample_offsets_job(cls, data_path: str, task: Tuple[int, Tuple[int, int]]) -> np.ndarray: 
        job_id, (start, end) = task  
        offsets = [start] 
        with open(data_path, mode='rb') as infile: 
            infile.seek(start, os.SEEK_SET) 
            with tqdm(total=None, desc=f'[Loacate Sample Offsets] job: {job_id}', position=job_id, disable=('DISABLE_TQDM' in os.environ)) as pbar: 
                while infile.tell() < end: 
                    infile.readline() 
                    offsets.append(infile.tell()) 
                    pbar.update()  
            assert offsets.pop() == end 
        offsets = np.array(offsets) 
        return offsets  

    @classmethod
    def _count_field_features_job(cls, data_path: str, task: Tuple[int, np.ndarray]) -> List[typing.Counter[bytes]]:
        job_id, sample_offsets = task  
        field_features_count = [Counter() for _ in range(Criteo.NUM_FIELDS)]
        with open(data_path, mode='rb') as infile:
            with tqdm(sample_offsets, desc=f'[Counting Field Features] job: {job_id}', position=job_id, disable=('DISABLE_TQDM' in os.environ)) as pbar: 
                for offset in pbar: 
                    infile.seek(offset) 
                    Criteo._count_one_line(infile.readline(), field_features_count) 
        return field_features_count 

    @classmethod 
    def _count_one_line(cls, line: bytes, field_features_count: List[typing.Counter[bytes]]) -> None: 
        label, *values = line.rstrip(b'\n').split(b'\t') 
        for field_id in Criteo.FIELDS_I: 
                field_features_count[field_id][Criteo._quantize_I_feature(values[field_id])] += 1 
        for field_id in Criteo.FIELDS_C: 
                field_features_count[field_id][values[field_id]] += 1    
    
    @classmethod 
    def _quantize_I_feature(cls, value: bytes) -> str: 
        if value == b'': 
            return 'NULL' 
        value = int(value) 
        if value > 2: 
            return str(int(math.log(value)**2)) 
        else: 
            return str(value - 2)
