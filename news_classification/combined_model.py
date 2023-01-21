import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score


class CombinedModel(torch.nn.Module):
    def __init__(self, n_hidden, n_class_output, n_layer=1):
        super(CombinedModel, self).__init__()
        self.hidden = torch.nn.Linear(768, n_hidden)  # hidden layer
        self.relu = nn.ReLU()
        layers = []
        for i in range(n_layer):
            layers.append(torch.nn.Linear(n_hidden, n_hidden))
        self.layers = torch.nn.ModuleList(layers)
        #self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_class_output)  # classification
        self.out2 = torch.nn.Linear(n_hidden, 1)  # regression

    def forward(self, x):
        x = self.relu(self.hidden(x))  # activation function for hidden layer
        for layer in self.layers:
            x = self.relu(layer(x))
        #x = nn.ReLU(self.hidden2(x))  # activation function for hidden layer
        x_out = F.softmax(self.out(x))
        x_out2 = self.out2(x)
        return x_out, x_out2

    def r2_loss(self, output, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2).float()
        ss_res = torch.sum((target - output) ** 2).float()
        r2 = 1 - ss_res / ss_tot
        return r2

    def learn(self, train_iter, dev_X, dev_y1, dev_y2, test_X, test_y1, test_y2, cfg):
        result = {}
        optimizer = optim.Adam(self.parameters(), lr=cfg.learning_rate)

        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
        loss_func2 = torch.nn.MSELoss()  # this is for regression mean squared loss

        all_train_loss = []
        all_dev_loss = []
        all_train_acc = []
        all_dev_acc = []
        all_train_r2 = []
        all_dev_r2 = []

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
                loss2 = loss_func2(y_out_2, batch_y2.unsqueeze(1))
                loss_total = loss1 + loss2
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                del loss1
                del loss2
                del loss_total

            train_correct = 0
            train_losses = 0
            train_all = 0
            train_r2_losses = 0

            # one train epoch finished, evaluate on the train and the dev set (NOT the test)
            for bi, (batch_x, batch_y1, batch_y2) in tqdm(enumerate(train_iter.iterate_once())):
                train_out_1, train_out_2 = self.forward(batch_x)
                train_loss_1 = loss_func(train_out_1, batch_y1)
                train_loss_2 = loss_func2(train_out_2, batch_y2.unsqueeze(1))
                train_loss = train_loss_1 + train_loss_2
                train_pred_1 = train_out_1.max(axis=1)[1]
                train_all += len(batch_x)
                train_correct += torch.eq(train_pred_1, batch_y1).sum().float()
                train_losses += len(batch_x) * train_loss.item()
                train_r2_losses += len(batch_x) * self.r2_loss(train_out_2.detach(), batch_y2.unsqueeze(1).detach())
                # train_acc_1 = float(torch.eq(train_pred_1, train_y1).sum().float() / len(train_X))
                del train_loss_1
                del train_loss_2
                del train_loss

            train_acc_1 = train_correct / train_all
            train_loss = float(train_losses) / train_all
            train_r2 = train_r2_losses / train_all
            all_train_acc.append(train_acc_1)
            all_train_loss.append(train_loss)
            all_train_r2.append(train_r2)

            dev_out_1, dev_out_2 = self.forward(dev_X)
            dev_loss_1 = loss_func(dev_out_1, dev_y1)
            dev_loss_2 = loss_func2(dev_out_2, dev_y2.unsqueeze(1))
            dev_loss = dev_loss_1 + dev_loss_2
            all_dev_loss.append(dev_loss.item())
            dev_pred = dev_out_1.max(axis=1)[1]
            dev_acc = float(torch.eq(dev_pred, dev_y1).sum().float() / len(dev_X))
            dev_r2 = self.r2_loss(dev_out_2.detach(), dev_y2.unsqueeze(1).detach())
            all_dev_acc.append(dev_acc)
            all_dev_r2.append(dev_r2)

            print(f"Epoch: {epoch}\n  train accuracy: {train_acc_1}  train loss: {train_loss}  r2: {train_r2}")
            print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss.item()}  r2: {dev_r2}")
            print(torch.cuda.memory_allocated())

            if min_loss - dev_loss.item() > 0.0001:
                epochs_no_improve = 0
                min_loss = dev_loss.item()
                best_epoch = epoch
                torch.save(self, os.path.join(cfg.training_dir, "model.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    early_stopping = True
                    print("Early stopping")
            if early_stopping:
                break

        test_out_1, test_out_2 = self.forward(test_X)
        test_loss_1 = loss_func(test_out_1, test_y1)
        test_loss_2 = loss_func(test_out_2, test_y2.unsqueeze(1))
        test_loss = test_loss_1 + test_loss_2
        test_pred = test_out_1.max(axis=1)[1]
        test_acc = float(torch.eq(test_pred, test_y1).sum().float() / len(test_X))
        test_loss = test_loss.item()
        test_r2 = self.r2_loss(test_out_2.detach(), test_y2.unsqueeze(1).detach())

        result["train_acc"] = all_train_acc[best_epoch]
        result["train_loss"] = all_train_loss[best_epoch]
        result['train_r2'] = all_train_r2[best_epoch]
        result["dev_acc"] = all_dev_acc[best_epoch]
        result["dev_loss"] = all_dev_loss[best_epoch]
        result['dev_r2'] = all_dev_r2[best_epoch]
        result["test_acc"] = test_acc
        result["test_loss"] = test_loss
        result['test_r2'] = test_r2
        result['epochs'] = best_epoch

        return result
