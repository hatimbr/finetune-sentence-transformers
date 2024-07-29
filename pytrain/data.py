from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class LegalPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        """Return the number of element of the dataset"""
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """Return the input for the model and the label for the loss"""
        df_elem = self.df.iloc[idx]
        query = df_elem["query"]
        output = df_elem["output"]
        return query, output


def collate_fn(
    batch: list[tuple[str, str]], tokenizer: PreTrainedTokenizer
) -> BatchEncoding:
    batch = [sentence for pairs in batch for sentence in pairs]
    model_inp = tokenizer(
        batch, padding=True, truncation=True, return_tensors='pt', max_length=128
    )
    return model_inp


def get_dataloader(
    parquet_path: Path, batch_size: int, tokenizer: PreTrainedTokenizer
) -> tuple[DataLoader, DataLoader]:
    train_dataframe = pd.read_parquet(parquet_path)
    valid_dataframe = train_dataframe.sample(frac=0.1, random_state=53)
    train_dataframe = train_dataframe.drop(valid_dataframe.index)

    train_dataset = LegalPairDataset(train_dataframe)
    valid_dataset = LegalPairDataset(valid_dataframe)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        prefetch_factor=1,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=1,
        prefetch_factor=1,
        drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    return train_dataloader, valid_dataloader
