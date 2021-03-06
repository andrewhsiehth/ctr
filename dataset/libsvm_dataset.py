from utils import fs  

import torch 
from torch.utils.data import Dataset 

from tqdm.auto import tqdm 

import numpy as np 

import io 
import os 
import itertools 
import functools 
import multiprocessing as mp 

from typing import List, Tuple  

class LIBSVMDataset(Dataset): 
    def __init__(self, data_uri: str, sample_offset: np.ndarray): 
        self.data_uri = data_uri  
        self.sample_offset = sample_offset 

    def __len__(self) -> int: 
        return len(self.sample_offset) 

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]: 
        with fs.open_buffered_file_reader(self.data_uri) as infile: 
            infile.seek(self.sample_offset[idx], io.SEEK_SET) 
            sample = infile.readline() 
        return LIBSVMDataset.parse_sample(sample) 

    @classmethod 
    def parse_sample(cls, sample: bytes) -> Tuple[torch.Tensor, torch.Tensor, int]: 
        label, *entries = sample.rstrip(b'\n').split(b' ') 
        feature_idx = torch.zeros(len(entries), dtype=torch.long)   
        feature_value = torch.zeros(len(entries), dtype=torch.float) 
        for i, entry in enumerate(entries): 
            fidx, fvalue = entry.split(b':') 
            feature_idx[i], feature_value[i] = int(fidx), float(fvalue) 
        return feature_idx, feature_value, int(label) 
        
    @classmethod 
    def prepare_dataset(cls, data_uri: str, n_jobs: int=os.cpu_count()): 
        sample_offset = LIBSVMDataset._locate_sample_offsets(data_uri=data_uri, n_jobs=n_jobs) 
        return LIBSVMDataset(data_uri=data_uri, sample_offset=sample_offset)  
        
    @classmethod 
    def _locate_sample_offsets(cls, data_uri: str, n_jobs: int) -> np.ndarray: 
        finfo = fs.file_info(data_uri) 
        chunk_size, _ = divmod(finfo.size, n_jobs) 
        
        chunk_starts = [0]
        with fs.open_buffered_file_reader(data_uri) as infile: 
            while infile.tell() < finfo.size: 
                infile.seek(chunk_size, os.SEEK_CUR) 
                infile.readline() 
                chunk_starts.append(min(infile.tell(), finfo.size)) 

        with mp.Pool(processes=n_jobs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool: 
            return np.asarray(list(itertools.chain.from_iterable(pool.imap(
                functools.partial(LIBSVMDataset._locate_sample_offsets_job, data_uri), 
                iterable=enumerate(zip(chunk_starts[:-1], chunk_starts[1:]))
            ))))

    @classmethod 
    def _locate_sample_offsets_job(cls, data_uri: str, task: Tuple[int, Tuple[int, int]]) -> List[int]: 
        job_id, (start, end) = task  
        offsets = [start] 
        with fs.open_buffered_file_reader(data_uri) as infile: 
            infile.seek(start, os.SEEK_SET) 
            with tqdm(total=None, desc=f'[Loacate Sample Offsets] job: {job_id}', position=job_id, disable=('DISABLE_TQDM' in os.environ)) as pbar: 
                while infile.tell() < end: 
                    infile.readline() 
                    offsets.append(infile.tell()) 
                    pbar.update()  
            assert offsets.pop() == end  
        return offsets  
