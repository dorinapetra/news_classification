import os

import numpy as np
import pandas as pd
from datasets import DatasetDict


def convert_dataset_to_jsonl(path):
    dataset = DatasetDict.load_from_disk(path)
    df = dataset['train'].to_pandas()
    train, validate, test = np.split(df.sample(frac=1, random_state=123),
                                     [int(0.7 * len(df)),
                                      int((0.7 + 0.15) * len(df))])
    train.to_json(os.path.join(path, 'train.jsonl'), force_ascii=False, lines=True, orient='records')
    validate.to_json(os.path.join(path, 'valid.jsonl'), force_ascii=False, lines=True, orient='records')
    test.to_json(os.path.join(path, 'test.jsonl'), force_ascii=False, lines=True, orient='records')


def load_train_valid_test(path):
    train = pd.read_json(os.path.join(path, 'train.jsonl'), lines=True)
    valid = pd.read_json(os.path.join(path, 'valid.jsonl'), lines=True)
    test = pd.read_json(os.path.join(path, 'test.jsonl'), lines=True)

    return train, valid, test
