import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from multi_head_attention import MultiHeadAttention
from multi_head_attention import PositionalEncoding

from transformers import BertModel


class ATT_ATT(nn.Module):
    def __init__(self, device, hps, loss_weights):
        super(ATT_ATT, self).__init__()
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

        self._positional_emb_1 = PositionalEncoding(self._emb_bert_dim)
        self._attention_1 = MultiHeadAttention(self._emb_bert_dim, self._hps['hidden1'], self._hps['heads'])
        self._norm_1 = nn.LayerNorm(self._hps['hidden1'])

        self._positional_emb_2 = PositionalEncoding(self._hps['hidden1'])
        self._attention_2 = MultiHeadAttention(self._hps['hidden1'], self._hps['hidden2'], self._hps['heads'])
        self._norm_2 = nn.LayerNorm(self._hps['hidden2'])

        self._fc = nn.Linear(self._hps['hidden2'], 3)

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

    def _pointwise_max(self, tensors):
        tensors = tensors.view(self._hps['msg_len'], -1, self._hps['batch_size'] * self._hps['conv_len'])
        t_prev = tensors[0]
        for t in tensors:
            t_prev = torch.max(t_prev, t)

        t_prev = t_prev.view(self._hps['batch_size'] * self._hps['conv_len'], -1)
        return t_prev

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
        return list(map(lambda x: np.argmax(x), preds.tolist()))

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
        msk_msg = msk_msg[:, :, 1:-1]

        bert_emb = self._dropout_emb(bert_emb)

        # first net
        if self._hps['positional']:
            bert_emb = self._positional_emb_1(bert_emb)

        att_msk = msk_msg.view(self._hps['batch_size'] * self._hps['conv_len'], self._hps['msg_len'])
        att_msk = att_msk \
            .unsqueeze(1).unsqueeze(-1) \
            .repeat(1, self._hps['heads'], 1, self._hps['msg_len'])  # batch*conv, heads, msg_len, msg_len
        att = self._attention_1(bert_emb, att_msk)
        att_emb = self._norm_1(att)

        att_emb = att_emb.view(self._hps['msg_len'], self._hps['batch_size'] * self._hps['conv_len'], -1)
        msk_msg = msk_msg.reshape(self._hps['msg_len'], self._hps['batch_size'] * self._hps['conv_len']).unsqueeze(
            -1)
        att_emb = att_emb * msk_msg

        dropped = self._dropout1(att_emb)
        first_net = self._pointwise_max(dropped)

        first_net = first_net.view(self._hps['batch_size'], self._hps['conv_len'], -1)

        # second net
        if self._hps['positional']:
            first_net = self._positional_emb_2(first_net)

        att_msk = msk_conv.view(self._hps['batch_size'], self._hps['conv_len'])
        att_msk = att_msk \
            .unsqueeze(1).unsqueeze(-1) \
            .repeat(1, self._hps['heads'], 1, self._hps['conv_len'])  # batch, heads, msg_len, msg_len
        att = self._attention_2(first_net, att_msk)
        second_net = self._norm_2(att)
        second_net = self._dropout2(second_net)

        second_net = second_net.view(self._hps['conv_len'], self._hps['batch_size'], -1)

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

            self._summary_writer.add_histogram('att_out_1/bias', self._attention_1.o_proj.bias, self._global_step)
            self._summary_writer.add_histogram('att_out_1/weight', self._attention_1.o_proj.weight, self._global_step)
            self._summary_writer.add_histogram('att_out_2/bias', self._attention_2.o_proj.bias, self._global_step)
            self._summary_writer.add_histogram('att_out_2/weight', self._attention_2.o_proj.weight, self._global_step)
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
            if self._global_step % self._hps['save'] == 0:
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
