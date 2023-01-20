import torch.nn as nn
import torch.optim as optim
from batched_iterator import BatchedIterator
import torch
import os
import numpy as np


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_value):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(dropout_value)

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop_out2 = nn.Dropout(dropout_value)
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, X):
        h = self.input_layer(X)
        h = self.relu(h)
        h = self.drop_out(h)

        h = self.hidden_layer(h)
        h = self.relu2(h)
        h = self.drop_out2(h)
        out = self.output_layer(h)
        return out

    def learn(self, train_X, train_y, dev_X, dev_y, epochs, batch_size):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())

        train_iter = BatchedIterator(train_X, train_y, batch_size)

        all_train_loss = []
        all_dev_loss = []
        all_train_acc = []
        all_dev_acc = []

        patience = 5
        epochs_no_improve = 0
        min_loss = np.Inf
        early_stopping = False
        best_epoch = 0

        for epoch in range(epochs):
            # training loop
            for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
                y_out = self.forward(batch_x)
                loss = criterion(y_out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # one train epoch finished, evaluate on the train and the dev set (NOT the test)
            train_out = self.forward(train_X)
            train_loss = criterion(train_out, train_y)
            all_train_loss.append(train_loss.item())
            train_pred = train_out.max(axis=1)[1]
            train_acc = float(torch.eq(train_pred, train_y).sum().float() / len(train_X))
            all_train_acc.append(train_acc)

            dev_out = self.forward(dev_X)
            dev_loss = criterion(dev_out, dev_y)
            all_dev_loss.append(dev_loss.item())
            dev_pred = dev_out.max(axis=1)[1]
            dev_acc = float(torch.eq(dev_pred, dev_y).sum().float() / len(dev_X))
            all_dev_acc.append(dev_acc)

            print(f"Epoch: {epoch}\n  train accuracy: {train_acc}  train loss: {train_loss}")
            print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")

            if min_loss - dev_loss > 0.001:
                epochs_no_improve = 0
                min_loss = dev_loss
                best_epoch = epoch
                torch.save(self, os.getcwd() + "/model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    early_stopping = True
                    print("Early stopping")
            if early_stopping:
                break
        return all_train_acc[best_epoch], all_train_loss[best_epoch], all_dev_acc[best_epoch], all_dev_loss[best_epoch]