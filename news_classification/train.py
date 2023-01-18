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

from batched_iterator import BatchedIterator
from classifier import SimpleClassifier


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



def learn(network, train_X, train_y, dev_X, dev_y, epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters())

    train_iter = BatchedIterator(train_X, train_y, batch_size)

    all_train_loss = []
    all_dev_loss = []
    all_train_acc = []
    all_dev_acc = []

    patience = 7
    epochs_no_improve = 0
    min_loss = np.Inf
    early_stopping = False
    best_epoch = 0

    for epoch in range(epochs):
        # training loop
        for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
            y_out = network(batch_x)
            loss = criterion(y_out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # one train epoch finished, evaluate on the train and the dev set (NOT the test)
        train_out = network(train_X)
        train_loss = criterion(train_out, train_y)
        all_train_loss.append(train_loss.item())
        train_pred = train_out.max(axis=1)[1]
        train_acc = float(torch.eq(train_pred, train_y).sum().float() / len(train_X))
        all_train_acc.append(train_acc)

        dev_out = network(dev_X)
        dev_loss = criterion(dev_out, dev_y)
        all_dev_loss.append(dev_loss.item())
        dev_pred = dev_out.max(axis=1)[1]
        dev_acc = float(torch.eq(dev_pred, dev_y).sum().float() / len(dev_X))
        all_dev_acc.append(dev_acc)

        print(f"Epoch: {epoch}\n  train accuracy: {train_acc}  train loss: {train_loss}")
        print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")

        if min_loss - dev_loss > 0.0001:
            epochs_no_improve = 0
            min_loss = dev_loss
            best_epoch = epoch
            torch.save(network, os.getcwd() + "/model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                early_stopping = True
                print("Early stopping")
        if early_stopping:
            break

    # TODO test
    return all_train_acc[best_epoch], all_train_loss[best_epoch], all_dev_acc[best_epoch], all_dev_loss[best_epoch]


@click.command()
@click.argument('config_file')
def main(config_file):
    cfg = get_config_from_yaml(config_file)

    result = {"start_time": datetime.now()}
    model_result = {}

    dataset, class_label = load_data()
    if cfg.load_tokenized_data:
        dataset = DatasetDict.load_from_disk(cfg.preprocessed_dataset_path)
    else:
        dataset, class_label = load_data()
        dataset.save_to_disk(cfg.preprocessed_dataset_path)

    train_X = torch.tensor(dataset['train'][cfg.input_name])
    dev_X = torch.tensor(dataset['validation'][cfg.input_name])
    train_y = torch.tensor(dataset['train'][cfg.output_name])
    dev_y = torch.tensor(dataset['validation'][cfg.output_name])

    model = SimpleClassifier(
        input_dim=train_X.size(1),
        output_dim=len(class_label.names),
        hidden_dim=cfg.hidden_dim,
        dropout_value=cfg.dropout
    )

    train_acc, train_loss, dev_acc, dev_loss = learn(model, train_X, train_y, dev_X, dev_y, epochs=cfg.epochs,
                                                     batch_size=cfg.batch_size)
    result["running_time"] = (datetime.now() - result["start_time"]).total_seconds()
    result["train_acc"] = train_acc
    result["train_loss"] = train_loss
    result["dev_acc"] = dev_acc
    result["dev_loss"] = dev_loss
    result["training"] = model_result
    with open(os.getcwd() + "/result.yaml", 'w+') as file:
        yaml.dump(result, file)


if __name__ == '__main__':
    main()
