import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from batched_iterator import BatchedIterator, BatchedIterator2


class CombinedModel(torch.nn.Module):
    def __init__(self, n_hidden, n_class_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(768, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_class_output)  # classification
        self.out2 = torch.nn.Linear(n_hidden, 1)  # regression

    def forward(self, x):
        x = nn.ReLU()(self.hidden(x))  # activation function for hidden layer
        x_out = F.softmax(self.out(x))
        x_out2 = self.out2(x)
        return x_out, x_out2

    def learn(self, train_X, train_y1, train_y2, dev_X, dev_y1, dev_y2, test_X, test_y, cfg):
        optimizer = optim.Adam(self.parameters())

        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
        loss_func2 = torch.nn.MSELoss()  # this is for regression mean squared loss

        train_iter = BatchedIterator2(train_X, train_y1, train_y2, cfg.batch_size)

        all_train_loss = []
        all_dev_loss = []
        all_train_acc = []
        all_dev_acc = []

        patience = cfg.patience
        epochs_no_improve = 0
        min_loss = np.Inf
        early_stopping = False
        best_epoch = 0

        for epoch in range(cfg.epochs):
            # training loop
            for bi, (batch_x, batch_y1, batch_y2) in tqdm(enumerate(train_iter.iterate_once())):
                y_out_1, y_out_2 = self.forward(batch_x)
                loss1 = loss_func(y_out_1, batch_y1)
                loss2 = loss_func2(y_out_2, batch_y2)
                loss_total = loss1 + loss2
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            # one train epoch finished, evaluate on the train and the dev set (NOT the test)
            train_out_1, train_out_2 = self.forward(train_X)
            train_loss_1 = loss_func(train_out_1, train_y1)
            train_loss_2 = loss_func2(train_out_2, train_y2)
            train_loss = train_loss_1 + train_loss_2
            all_train_loss.append(train_loss.item())
            train_pred_1 = train_out_1.max(axis=1)[1]
            train_acc_1 = float(torch.eq(train_pred_1, train_y1).sum().float() / len(train_X))
            all_train_acc.append(train_acc_1)

            dev_out_1, dev_out_2 = self.forward(dev_X)
            dev_loss_1 = loss_func(dev_out_1, dev_y1)
            dev_loss_2 = loss_func2(dev_out_2, dev_y2)
            dev_loss = dev_loss_1 + dev_loss_2
            all_dev_loss.append(dev_loss.item())
            dev_pred = dev_out_1.max(axis=1)[1]
            dev_acc = float(torch.eq(dev_pred, dev_y1).sum().float() / len(dev_X))
            all_dev_acc.append(dev_acc)

            print(f"Epoch: {epoch}\n  train accuracy: {train_acc_1}  train loss: {train_loss}")
            print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")

            if min_loss - dev_loss > 0.0001:
                epochs_no_improve = 0
                min_loss = dev_loss
                best_epoch = epoch
                torch.save(self, os.path.join(cfg.training_dir + "/model.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    early_stopping = True
                    print("Early stopping")
            if early_stopping:
                break

        # test_out = self.forward(test_X)
        # test_loss = criterion(test_out, test_y)
        # test_pred = test_out.max(axis=1)[1]
        # test_acc = float(torch.eq(test_pred, test_y).sum().float() / len(test_X))
        # test_loss_v = test_loss.item()

        test_acc = 0
        test_loss_v = 0

        return all_train_acc[best_epoch], all_train_loss[best_epoch], all_dev_acc[best_epoch], all_dev_loss[
            best_epoch], test_acc, test_loss_v
