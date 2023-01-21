import os
from datetime import datetime

import click
import numpy as np
import torch
import torch.optim as optim
import yaml
from datasets import load_dataset, DatasetDict
from dotmap import DotMap
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from batched_iterator import BatchedIterator, BatchedIterator2
from classifier import SimpleClassifier
from combined_model import CombinedModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.FullLoader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config


def _process_data(batch):
    batch['year'] = batch['date_of_creation'].year
    batch['label'] = str(batch['year']) + '-' + batch['domain']

    return batch

def tokenize_data(batch):
    inputs = tokenizer(batch['article'], padding='max_length', truncation=True, max_length=512)
    input_ids = torch.tensor(inputs.input_ids).to('cuda')
    attention_mask = torch.tensor(inputs.attention_mask).to('cuda')
    output = bert_model(input_ids=torch.tensor(input_ids),
                   attention_mask=torch.tensor(attention_mask))
    batch['cls_token'] = list(output.pooler_output)
    batch['start_token'] = list(output.last_hidden_state[:, 0, :])
    batch['avg_token'] = list(torch.mean(output.last_hidden_state, dim=1))

    return batch

tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
bert_model = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
bert_model.eval()

def load_data(descartes=True):
    dataset = load_dataset("SZTAKI-HLT/HunSum-1")
    dataset = dataset.remove_columns(['title', 'lead', 'tags', 'url'])
    dataset = dataset.filter(lambda x: x["date_of_creation"] != None)

    bert_model.to('cuda')
    dataset = dataset.map(lambda x: _process_data(x), batched=False)
    dataset = dataset.map(lambda x: tokenize_data(x), batched=True, batch_size=50)
    if descartes:
        dataset = dataset.class_encode_column("label")
        class_label = dataset['train'].features['label']
    else:
        # TODO
        dataset = dataset.class_encode_column('domain')
        dataset = dataset.class_encode_column('year')
    return dataset, class_label




@click.command()
@click.argument('config_file')
def main(config_file):
    cfg = get_config_from_yaml(config_file)

    result = {"start_time": datetime.now()}

    if cfg.load_tokenized_data:
        dataset = DatasetDict.load_from_disk(cfg.preprocessed_dataset_path).remove_columns(['date_of_creation']).with_format("torch", device=device)
    else:
        dataset, class_label = load_data()
        dataset.save_to_disk(cfg.preprocessed_dataset_path)

    dataset = dataset.class_encode_column('domain')

    classes = dataset['train'].features['domain'].names

    train_X = dataset['train'][cfg.input_name]
    dev_X = dataset['validation'][cfg.input_name]
    test_X = dataset['test'][cfg.input_name]
    train_y1 = dataset['train']['domain']
    train_y2 = dataset['train']['year']
    dev_y1 = dataset['validation']['domain']
    dev_y2 = dataset['validation']['year']
    test_y1 = dataset['test']['domain']
    test_y2 = dataset['test']['year']

    min_year = train_y2.min()
    max_year = train_y2.max()

    train_y2 -= min_year
    train_y2 = (train_y2.type(torch.FloatTensor).to(device) / (max_year - min_year)) * 30

    dev_y2 -= min_year
    dev_y2 = (dev_y2.type(torch.FloatTensor).to(device) / (max_year - min_year)) * 30

    test_y2 -= min_year
    test_y2 = (test_y2.type(torch.FloatTensor).to(device) / (max_year - min_year)) * 30

    model = CombinedModel(cfg.hidden_dim, len(classes), n_layer=cfg.n_layer).to(device)

    train_iter = BatchedIterator2(train_X, train_y1, train_y2, cfg.batch_size)

    train_result = model.learn(train_iter, dev_X, dev_y1, dev_y2, test_X, test_y1, test_y2, cfg)
    result["running_time"] = (datetime.now() - result["start_time"]).total_seconds()
    result['num_classes'] = len(classes)
    result['min_year'] = min_year.float()
    result['max_year'] = max_year.float()
    result['classes'] = classes
    result["train_result"] = train_result
    result["learning_rate"] = cfg.learning_rate
    result["hidden_dim"] = cfg.hidden_dim
    result["n_layer"] = cfg.n_layer

    with open(os.path.join(cfg.training_dir, "result.yaml"), 'w+') as file:
        yaml.dump(result, file)


if __name__ == '__main__':
    main()
