import os
from datetime import datetime

import click
import numpy as np
import torch
import torch.optim as optim
import yaml
from datasets import load_dataset
from dotmap import DotMap
from torch import nn
from transformers import AutoTokenizer, AutoModel

from news_classification.batched_iterator import BatchedIterator
from news_classification.classifier import SimpleClassifier


def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.FullLoader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config


def _process_data_target(batch):
    batch['year'] = batch['date_of_creation'].year
    batch['label'] = str(batch['year']) + '-' + batch['domain']
    return batch


def load_data(descartes=True):
    dataset = load_dataset("SZTAKI-HLT/HunSum-1")
    dataset = dataset.remove_columns(['title', 'lead', 'tags', 'url'])
    dataset = dataset.filter(lambda x: x["date_of_creation"] != None)
    dataset = dataset.map(_process_data_target, batched=False)
    if descartes:
        dataset = dataset.class_encode_column("label")
        class_label = dataset['train'].features['label']
    else:
        # TODO
        dataset = dataset.class_encode_column('domain')
        dataset = dataset.class_encode_column('year')
    return dataset, class_label


def get_expected_output(dataset, output_vocab):
    data = dataset.to_pandas()
    y = torch.empty(len(data))
    for i in range(len(data)):
        y[i] = output_vocab[data[3][i].rstrip("\n")]
    y = y.type(torch.LongTensor)
    return y


def get_encoded_wordpieces(dataset, model, tokenizer):
    batch_size = 50
    data = dataset.to_pandas()
    X = torch.empty([len(data), 768])
    batch_iter = BatchedIterator(data['article'].tolist(), data['label'].tolist(), batch_size)
    for bi, (batch_x, batch_y) in enumerate(batch_iter.iterate_once()):
        tokens = []
        index = []
        for i, article in enumerate(batch_x):
            article_tokens = tokenizer(article)['input_ids']
            tokens.extend([article_tokens])
            index.extend([0])
        # padding
        length = max(map(len, tokens))
        padded = torch.tensor([xi + [0] * (length - len(xi)) for xi in tokens])
        # masking
        mask = torch.where(padded > 0, torch.ones(padded.shape), torch.zeros(padded.shape))
        with torch.no_grad():
            output = model(input_ids=padded, attention_mask=mask)
        # X[bi * batch_size:bi * batch_size + len(batch_x)] = output[0][np.arange(len(output[0])), index]
        X[bi * batch_size:bi * batch_size + len(batch_x)] = output[1]
    return X


def get_bert_output(dataset):
    tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
    bert_model = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
    bert_model.eval()
    train_X = get_encoded_wordpieces(dataset['train'], bert_model, tokenizer)
    # train_y = get_expected_output(dataset['train'], class_label)
    train_y = dataset['train']['label']
    dev_X = get_encoded_wordpieces(dataset['validation'], bert_model, tokenizer)
    # dev_y = get_expected_output(dataset['validation'], class_label)
    dev_y = dataset['validation']['label']
    return train_X, train_y, dev_X, dev_y


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

    train_X, train_y, dev_X, dev_y = get_bert_output(dataset)

    model = SimpleClassifier(
        input_dim=train_X.size(1),
        output_dim=len(class_label),
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
