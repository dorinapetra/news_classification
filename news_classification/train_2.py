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

from batched_iterator import BatchedIterator
from classifier import SimpleClassifier
from combined_model import CombinedModel


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
    model_result = {}

    if cfg.load_tokenized_data:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = DatasetDict.load_from_disk(cfg.preprocessed_dataset_path).remove_columns(['date_of_creation']).with_format("torch", device=device)
    else:
        dataset, class_label = load_data()
        dataset.save_to_disk(cfg.preprocessed_dataset_path)
    
    # dataset = dataset.filter(lambda x: x["date_of_creation"] > datetime(2003, 1, 1))
    # dataset = dataset.filter(lambda x: x["date_of_creation"] < datetime(2023, 1, 1))
    # dataset = dataset.filter(lambda x: x["domain"] != "telex.hu")
    # dataset = dataset.filter(lambda x: x["domain"] != "metropol.hu")

    dataset = dataset.class_encode_column('domain')

    # train_X = torch.tensor(dataset['train'][cfg.input_name]).to('cuda')
    # dev_X = torch.tensor(dataset['validation'][cfg.input_name]).to('cuda')
    # test_X = torch.tensor(dataset['test'][cfg.input_name]).to('cuda')
    # train_y1 = torch.tensor(dataset['train']['domain']).to('cuda')
    # train_y2 = torch.tensor(dataset['train']['year']).to('cuda')
    # dev_y1 = torch.tensor(dataset['validation']['domain']).to('cuda')
    # dev_y2 = torch.tensor(dataset['validation']['year']).to('cuda')
    # test_y = torch.tensor(dataset['test']['domain']).to('cuda')

    train_X = dataset['train'][cfg.input_name]
    dev_X = dataset['validation'][cfg.input_name]
    test_X = dataset['test'][cfg.input_name]
    train_y1 = dataset['train']['domain']
    train_y2 = dataset['train']['year']
    dev_y1 = dataset['validation']['domain']
    dev_y2 = dataset['validation']['year']
    test_y = dataset['test']['domain']

    train_y2 -= train_y2.min(1, keepdim=True)[0]
    train_y2 /= train_y2.max(1, keepdim=True)[0]

    dev_y2 -= train_y2.min(1, keepdim=True)[0]
    dev_y2 /= train_y2.max(1, keepdim=True)[0]



    # model = SimpleClassifier(
    #     input_dim=train_X.size(1),
    #     output_dim=len(class_label.names),
    #     hidden_dim=cfg.hidden_dim,
    #     dropout_value=cfg.dropout
    # )

    model = CombinedModel(100, 9).to('cuda')

    train_acc, train_loss, dev_acc, dev_loss, test_acc, test_loss = model.learn(train_X, train_y1, train_y2, dev_X, dev_y1, dev_y2, test_X, test_y, cfg)
    result["running_time"] = (datetime.now() - result["start_time"]).total_seconds()
    result["train_acc"] = train_acc
    result["train_loss"] = train_loss
    result["dev_acc"] = dev_acc
    result["dev_loss"] = dev_loss
    result["test_acc"] = test_acc
    result["test_loss"] = test_loss
    result["training"] = model_result
    with open(os.path.join(cfg.training_dir , "/result.yaml"), 'w+') as file:
        yaml.dump(result, file)


if __name__ == '__main__':
    main()
