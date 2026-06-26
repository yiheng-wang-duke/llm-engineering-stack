import os
import random
import glob
from typing import Literal, List
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
from .read_jsonl import *



class LeetCodeDataset(Dataset):
    def __init__(self, path: str, split: Literal["test", "train"]):
        super().__init__()
        self.split = split
        if split=="train":
            # jsonl_files = glob.glob(os.path.join(path, f"LeetCodeDataset-train-cot-clean.jsonl"))
            jsonl_files = glob.glob(os.path.join(path, f"LeetCodeDataset-train.jsonl"))
        else:
            jsonl_files = glob.glob(os.path.join(path, f"LeetCodeDataset-test.jsonl"))
            
        self.data = []
        for f in jsonl_files:
            self.data.extend(read_jsonl(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

class OpenR1MathDataset(Dataset):
    """OpenR1-Math-220k dataset (parquet format from HuggingFace).

    Each sample returns the `messages` field: list[dict] with role/content.
    Supports subset: "all", "default", "extended".
    """

    def __init__(self, path: str, subset: str = "all"):
        super().__init__()
        hf_dataset = load_dataset(path, name=subset, split="train")
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["messages"]


class MixedDataset(Dataset):
    def __init__(self, datasets: List[Dataset], proportions: List[float], total_size: int):
        assert len(datasets) == len(proportions)
        assert abs(sum(proportions) - 1.0) < 1e-6

        self.indices = []  # list of (dataset_idx, sample_idx)
        for ds_idx, (ds, prop) in enumerate(zip(datasets, proportions)):
            n_samples = int(total_size * prop)
            sampled = [random.randint(0, len(ds) - 1) for _ in range(n_samples)]
            self.indices.extend([(ds_idx, i) for i in sampled])

        random.shuffle(self.indices)
        self.datasets = datasets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ds_idx, sample_idx = self.indices[index]
        return self.datasets[ds_idx][sample_idx]
    
def build_mixed_dataloader(
    datasets: List[Dataset],
    proportions: List[float],
    total_size: int,
    batch_size: int = 32,
    num_workers: int = 8,
) -> DataLoader:
    mixed = MixedDataset(datasets, proportions, total_size)
    return DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,  # 不做 collate，直接返回 list[dict]
    )
