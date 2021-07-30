import os

import numpy as np
import math
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from transformers import BertModel


class CNN_MLP(nn.Module):
    def __init__(self, device, hps, loss_weights):
        super(CNN_MLP, self).__init__()
        self._hps = hps
        self._device = device
        self._loss_weight = loss_weights

        now = datetime.now()
        dt_str = now.strftime("%d-%m-%Y-%H-%M-%S")
        comment = '__'.join([k + '_' + str(v) for k, v in self._hps.items()])

        path = os.path.join(self._hps["path"], 'runs')
        if not os.path.exists(path):
            os.makedirs(path)

        self._summary_writer = SummaryWriter(os.path.join(path, dt_str), comment=comment)

        self._define_model()
        self._optimizer = self._define_optimizer()

        self._global_step = 0

    def _define_model(self):
        self._bert = BertModel.from_pretrained(self._hps['bert'], output_hidden_states=True)

        self._emb_bert_dim = 768 * (4 if self._hps['layers'] == 'concat' else 1)
        self._fc_emb = nn.Linear(self._emb_bert_dim, self._hps['emb_dim'])
        self._fc_emb_relu = nn.ReLU()

        self._kernel_1 = 2
        self._kernel_2 = 3
        self._kernel_3 = 4
        self._kernel_4 = 5

        self._layer1 = nn.Sequential(
            nn.Conv1d(self._hps['msg_len'], self._hps['hidden1'], kernel_size=self._kernel_1,
                      stride=self._hps['stride']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self._kernel_1, stride=self._hps['stride'])
        )

        self._layer2 = nn.Sequential(
            nn.Conv1d(self._hps['msg_len'], self._hps['hidden1'], kernel_size=self._kernel_2,
                      stride=self._hps['stride']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self._kernel_2, stride=self._hps['stride'])
        )

        self._layer3 = nn.Sequential(
            nn.Conv1d(self._hps['msg_len'], self._hps['hidden1'], kernel_size=self._kernel_3,
                      stride=self._hps['stride']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self._kernel_3, stride=self._hps['stride'])
        )

        self._layer4 = nn.Sequential(
            nn.Conv1d(self._hps['msg_len'], self._hps['hidden1'], kernel_size=self._kernel_4,
                      stride=self._hps['stride']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self._kernel_4, stride=self._hps['stride'])
        )

        self._net_2 = nn.Sequential(
            nn.Linear(self._features_size(), self._hps['hidden2']),
            nn.ReLU(),
            nn.Linear(self._hps['hidden2'], self._hps['hidden3']),
            nn.ReLU(),
        )

        self._fc = nn.Linear(self._hps['hidden3'], 2)

        self._dropout_emb = nn.Dropout(self._hps['prob_emb'])
        self._dropout1 = nn.Dropout(self._hps['prob1'])
        self._dropout2 = nn.Dropout(self._hps['prob2'])
        self._softmax = nn.LogSoftmax(dim=1)

        self._criterion = nn.NLLLoss(ignore_index=-1, reduction='mean', weight=self._loss_weight)

    def _define_optimizer(self):
        opt = torch.optim.SGD(self.parameters(), self._hps['lr'])

        if self._hps['optimizer'] == 'ADAM':
            opt = torch.optim.Adam(self.parameters(), self._hps['lr'], weight_decay=self._hps['weight'])
        elif self._hps['optimizer'] == 'Adadelta':
            opt = torch.optim.Adadelta(self.parameters(), self._hps['lr'], weight_decay=self._hps['weight'])
        elif self._hps['optimizer'] == 'Adagrad':
            opt = torch.optim.Adagrad(self.parameters(), self._hps['lr'], weight_decay=self._hps['weight'])
        elif self._hps['optimizer'] == 'RSMProp':
            opt = torch.optim.RMSprop(self.parameters(), self._hps['lr'], weight_decay=self._hps['weight'])

        return opt

    def _features_size(self):
        out_conv_1 = math.floor(((self._hps['emb_dim'] - 1 * (self._kernel_1 - 1) - 1) / self._hps['stride']) + 1)
        out_pool_1 = math.floor(((out_conv_1 - 1 * (self._kernel_1 - 1) - 1) / self._hps['stride']) + 1)

        out_conv_2 = math.floor(((self._hps['emb_dim'] - 1 * (self._kernel_2 - 1) - 1) / self._hps['stride']) + 1)
        out_pool_2 = math.floor(((out_conv_2 - 1 * (self._kernel_2 - 1) - 1) / self._hps['stride']) + 1)

        out_conv_3 = math.floor(((self._hps['emb_dim'] - 1 * (self._kernel_3 - 1) - 1) / self._hps['stride']) + 1)
        out_pool_3 = math.floor(((out_conv_3 - 1 * (self._kernel_3 - 1) - 1) / self._hps['stride']) + 1)

        out_conv_4 = math.floor(((self._hps['emb_dim'] - 1 * (self._kernel_4 - 1) - 1) / self._hps['stride']) + 1)
        out_pool_4 = math.floor(((out_conv_4 - 1 * (self._kernel_4 - 1) - 1) / self._hps['stride']) + 1)

        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self._hps['hidden1']

    def _extract_layer(self, hidden_states):
        if self._hps['layers'] == 'last':
            return hidden_states[-1]
        elif self._hps['layers'] == 'second':
            return hidden_states[-2]
        elif self._hps['layers'] == 'sum_all':
            return torch.sum(torch.stack(hidden_states[1:]), dim=0)  # exclude first layer (embedding)
        elif self._hps['layers'] == 'sum_four':
            return torch.sum(torch.stack(hidden_states[-4:]), dim=0)
        elif self._hps['layers'] == 'concat':
            return torch.cat(hidden_states[-4:], dim=2)
        else:
            return hidden_states[-1]

    def _decode_softmax(self, pred, msk_conv):
        pred = pred.view(self._hps['batch_size'] * self._hps['conv_len'], -1)
        msk_conv = msk_conv.view(self._hps['batch_size'] * self._hps['conv_len'])
        indeces = torch.nonzero(msk_conv, as_tuple=True)
        preds = pred[indeces]
        return list(map(lambda x: np.argmax(x), preds.tolist())), preds.tolist()

    def close_writer(self):
        self._summary_writer.close()

    def get_states(self):
        return self.state_dict(), self._optimizer.state_dict(), self._global_step

    def load_state(self, checkpoint):
        self.load_model(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._global_step = checkpoint['step']

    def forward(self, x, msk_conv, msk_msg):
        """

        :param x: (batch, conv_len, msg_len+2)
        :param msk_conv: (batch, conv_len)
        :param msk_msg: (batch, conv_len, msg_len+2)
        :return: (conv_len, batch, classes)
        """

        input_ids = x.view(self._hps['batch_size'] * self._hps['conv_len'], -1)  # batch * conv_len, sequence_length
        attention_mask = msk_msg.view(self._hps['batch_size'] * self._hps['conv_len'], -1)

        self._bert.eval()
        with torch.no_grad():
            bert_emb_out = self._bert(input_ids, attention_mask)

        # word embeddings
        bert_emb_states = bert_emb_out[2]  # (batch * conv_len, sequence_length, hidden_size) for each layer (13)
        bert_emb = self._extract_layer(bert_emb_states)
        bert_emb = bert_emb[:, 1:-1, :]  # discard special tokens

        if self._hps['emb_dim'] != self._emb_bert_dim:
            bert_emb = self._fc_emb_relu(self._fc_emb(bert_emb))  # reduce dimensions

        embed_x = self._dropout_emb(bert_emb)

        # reshape x
        embed_x = embed_x.reshape(self._hps['batch_size'] * self._hps['conv_len'],  # batch * conv_len
                                  self._hps['msg_len'],  # msg_len
                                  self._hps['emb_dim'])  # hid_dim

        # first net
        x1 = self._layer1(embed_x)  # batch*conv, hidden, features
        x2 = self._layer2(embed_x)
        x3 = self._layer3(embed_x)
        x4 = self._layer4(embed_x)

        union = torch.cat((x1, x2, x3, x4), dim=2)  # batch, hidden, features
        union = union.reshape(self._hps['batch_size'] * self._hps['conv_len'], -1)  # flattening

        dropped = self._dropout1(union)

        first_net = dropped.view(self._hps['conv_len'], self._hps['batch_size'], self._hps['hidden2'])

        # second net
        net_out = self._net_2(first_net)  # conv_len, batch, hidden
        second_net = self._dropout2(net_out)

        # prediction
        msgs = []
        for msg in second_net:
            out = self._fc(msg)  # batch, classes
            out = self._softmax(out)
            msgs.append(out)

        output = torch.stack(msgs)  # conv_len, batch, classes
        msk_conv = msk_conv.view(self._hps['conv_len'], self._hps['batch_size']).unsqueeze(-1)
        output = output * msk_conv

        return output

    def fit(self, x, y, msk_conv, msk_msg):
        """
        Train the model

        :param x: input sequence (batch, conv_len, msg_len+2)
        :param y: target sequence (batch, conv_len)
        :param msk_conv: conversation mask (batch, conv_len)
        :param msk_msg: message mask (batch, conv_len, msg_len+2)
        :return: loss value, step
        """

        self.train()
        self._optimizer.zero_grad()

        preds = self(x, msk_conv, msk_msg)  # conv_len, batch, classes

        # compute average loss
        avg_loss = []
        pred_y = preds.view(self._hps['batch_size'], self._hps['conv_len'], -1)
        true_y = y.view(self._hps['batch_size'], self._hps['conv_len'])
        for i in range(self._hps['batch_size']):
            avg_loss.append(self._criterion(pred_y[i], true_y[i]))

        loss = torch.mean(torch.stack(avg_loss))
        loss_value = loss.item()

        # optimization step
        loss.backward()
        if self._hps['clip'] != -1:
            nn.utils.clip_grad_norm_(self.parameters(), self._hps['clip'])
        self._optimizer.step()

        # compute metrics
        if self._global_step % self._hps['save'] == 0:
            y_pred = self._decode_softmax(preds, msk_conv)
            y_test = y.view(self._hps['batch_size'] * self._hps['conv_len']).tolist()
            y_test = list(filter(lambda z: z != -1, y_test))  # ignore padding

            parameters = [p for p in self.parameters() if p.grad is not None]
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach()).to(self._device) for p in parameters]))  # L2 norm
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            self._summary_writer.add_scalar('Loss/train', loss_value, self._global_step)
            self._summary_writer.add_scalar('Precision/train', prec, self._global_step)
            self._summary_writer.add_scalar('Recall/train', rec, self._global_step)
            self._summary_writer.add_scalar('Grad norm/train', total_norm, self._global_step)

            if self._hps['emb_dim'] != self._emb_bert_dim:
                self._summary_writer.add_histogram('fc_emb/bias', self._fc_emb.bias, self._global_step)
                self._summary_writer.add_histogram('fc_emb/weight', self._fc_emb.weight, self._global_step)

            self._summary_writer.add_histogram('net_2[0]/bias', self._net_2[0].bias, self._global_step)
            self._summary_writer.add_histogram('net_2[0]/weight', self._net_2[0].weight, self._global_step)
            self._summary_writer.add_histogram('net_2[2]/bias', self._net_2[2].bias, self._global_step)
            self._summary_writer.add_histogram('net_2[2]/weight', self._net_2[2].weight, self._global_step)
            self._summary_writer.add_histogram('fc/bias', self._fc.bias, self._global_step)
            self._summary_writer.add_histogram('fc/weight', self._fc.weight, self._global_step)

            self._summary_writer.flush()

        self._global_step += 1

        return loss_value, self._global_step

    def valid(self, x, y, msk_conv, msk_msg):
        """
        Validate the model

        :param x: input sequence (batch, conv_len, msg_len+2)
        :param y: target sequence (batch, conv_len)
        :param msk_conv: conversation mask (batch, conv_len)
        :param msk_msg: message mask (batch, conv_len, msg_len+2)
        :return: loss value, step
        """

        with torch.no_grad():
            self.eval()
            preds = self(x, msk_conv, msk_msg)  # conv_len, batch, classes

            # compute average loss
            avg_loss = []
            pred_y = preds.view(self._hps['batch_size'], self._hps['conv_len'], -1)
            true_y = y.view(self._hps['batch_size'], self._hps['conv_len'])
            for i in range(self._hps['batch_size']):
                avg_loss.append(self._criterion(pred_y[i], true_y[i]))

            loss = torch.mean(torch.stack(avg_loss))
            loss_value = loss.item()

            # compute metrics
            y_pred = self._decode_softmax(preds, msk_conv)
            y_test = y.view(self._hps['batch_size'] * self._hps['conv_len']).tolist()
            y_test = list(filter(lambda z: z != -1, y_test))  # ignore padding

            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            self._summary_writer.add_scalar('Loss/valid', loss_value, self._global_step)
            self._summary_writer.add_scalar('Precision/valid', prec, self._global_step)
            self._summary_writer.add_scalar('Recall/valid', rec, self._global_step)
            self._summary_writer.flush()

            self._global_step += 1

            return loss_value, self._global_step

    def predict(self, x, msk_conv, msk_msg, no_batch=False):
        """
        Use the model for prediction

        :param x: input sequence (batch, conv_len, msg_len+2)
        :param msk_conv: conversation mask (batch, conv_len)
        :param msk_msg: message mask (batch, conv_len, msg_len+2)
        :param no_batch: true if there is only 1 batch
        :return: [unpad_conv_len]
        """

        if no_batch:
            self._hps['batch_size'] = 1

        with torch.no_grad():
            self.eval()
            preds = self(x, msk_conv, msk_msg)  # conv_len, batch, classes
            return self._decode_softmax(preds, msk_conv)
