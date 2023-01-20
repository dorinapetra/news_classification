import os

import numpy as np
from datasets import DatasetDict
import pandas as pd

def convert_to_jsonl(cfg):
    dataset = DatasetDict.load_from_disk(cfg.preprocessed_dataset_path)
    df = dataset.to_pandas()
    train, validate, test = np.split(df.sample(frac=1, random_state=123),
                                     [int(0.7 * len(df)),
                                      int((0.7 + 0.15) * len(df))])
    train.to_json(os.path.join(cfg.preprocessed_dataset_path, 'train.jsonl'), force_ascii=False, lines=True, orient='records')
    validate.to_json(os.path.join(cfg.preprocessed_dataset_path, 'valid.jsonl'), force_ascii=False, lines=True, orient='records')
    test.to_json(os.path.join(cfg.preprocessed_dataset_path, 'test.jsonl'), force_ascii=False, lines=True, orient='records')

def read_train_valid_test(cfg):
    train = pd.read_json(os.path.join(cfg.preprocessed_dataset_path, 'train.jsonl'), lines=True)
    valid = pd.read_json(os.path.join(cfg.preprocessed_dataset_path, 'valid.jsonl'), lines=True)
    test = pd.read_json(os.path.join(cfg.preprocessed_dataset_path, 'test.jsonl'), lines=True)

    return train, valid, test
