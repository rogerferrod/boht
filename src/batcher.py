from tensorflow.core.example import example_pb2
import random
from queue import Queue
from transformers import BertTokenizer

import torch
import glob
import struct
import numpy as np


class Batcher(object):
    def __init__(self, input, hps):
        self.input = input
        self._hps = hps

        self._tokenizer = BertTokenizer.from_pretrained(self._hps['bert'])
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = Queue()
        self._example_list = []
        self._finished_reading = False  # this will tell us when we're finished reading the dataset

        self.fill_example_queue()
        self.reset()

    def fill_example_queue(self):
        input_gen = text_generator(data_generator(self.input))

        for conv_id, conversation, target in input_gen:
            conversation_sentences = [sent.strip() for sent in conversation.split('</s>')]
            example = Conversation(conv_id, conversation_sentences, target, self._tokenizer, self._hps)
            self._example_list.append(example)

        self._example_list_len = len(self._example_list)

    def reset(self):
        random.Random(42).shuffle(self._example_list)

        if self._hps['mode'] == 'train':
            batches = []
            for i in range(0, self._example_list_len, self._hps['batch_size']):
                batches.append(self._example_list[i:i + self._hps['batch_size']])

            random.Random(42).shuffle(batches)
            for batch in batches:
                self._batch_queue.put(Batch(batch, self._device, self._hps))

        else:
            for idx in range(self._example_list_len):
                ex = self._example_list[idx]
                batch = [ex]
                self._batch_queue.put(Batch(batch, self._device, self._hps))

    def __next__(self):
        if self._batch_queue.qsize() == 0:
            self.reset()
            raise StopIteration

        return self._batch_queue.get()

    def __iter__(self):
        return self


class Sentence(object):
    def __init__(self, sent, tokenizer, hps):
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'

        self.hps = hps
        self.original_sent = sent

        sent = sent.strip().split(' ')
        self.original_sent_len = len(sent)

        if len(sent) > hps['msg_len']:
            sent = sent[:hps['msg_len']]

        self.sent_len = len(sent)

        self.sent = [self.cls_token] + sent

        # pad sentence
        self.pad_sent()

        # add the SEP tag
        self.sent += [self.sep_token]

        # tokenize
        self.sent = [tokenizer.convert_tokens_to_ids(s) for s in self.sent]

    def pad_sent(self):
        while len(self.sent) < self.hps['msg_len'] + 1:  # +1 to add the SEP token
            self.sent.append(self.pad_token)


class Conversation(object):
    def __init__(self, conv_id, sentences, target, tokenizer, hps):

        self.original_sentences = sentences
        self.hps = hps
        self.conv_id = conv_id
        self._tokenizer = tokenizer

        self._original_sentences_len = len(sentences)
        if len(sentences) > hps['conv_len']:
            sentences = sentences[:hps['conv_len']]

        self.target = self.set_target(target)

        sents = []
        for s in sentences:
            s = Sentence(s, self._tokenizer, hps)
            sents.append(s)

        self.sents = sents
        self.sents_len = len(sents)

    def pad_conversation(self):
        while len(self.sents) < self.hps['conv_len']:
            s = Sentence('', self._tokenizer, self.hps)
            self.sents.append(s)

    def set_target(self, target):
        target_list = list(map(lambda x: int(x), target.split('</s>')))
        target_list = target_list[:self.hps['conv_len']]
        t = [-1 for _ in range(self.hps['conv_len'])]
        for i, v in enumerate(target_list):
            t[i] = v
        return t


class Batch(object):
    def __init__(self, example_list, device, hps):
        self.hps = hps

        if len(example_list) > hps['batch_size']:
            example_list = example_list[:hps['batch_size']]

        convs = np.zeros((hps['batch_size'], hps['conv_len'], hps['msg_len'] + 2),
                         dtype=np.int)
        conv_mask = np.zeros((hps['batch_size'], hps['conv_len']),
                             dtype=np.float)
        sent_mask = np.zeros((hps['batch_size'], hps['conv_len'], hps['msg_len'] + 2),
                             dtype=np.float)
        target = -np.ones((hps['batch_size'], hps['conv_len']), dtype=np.int)

        for i, conv in enumerate(example_list):
            conv.pad_conversation()
            target[i, :] = conv.target[:]
            for j, s in enumerate(conv.sents):
                convs[i, j, :] = s.sent[:]
                if j < conv.sents_len:
                    conv_mask[i, j] = 1
                    for k in range(1, s.sent_len):  # from CLS (excluded) to latest word (no PAD)
                        sent_mask[i, j, k] = 1
                    sent_mask[i, j, 0] = sent_mask[i, j, -1] = 1  # CLS and SEP

        if self.hps['mode'] != 'train':  # single batch
            self.id = example_list[0].conv_id
            self.original = list(map(lambda x: x.original_sentences, example_list))[0]

        self.x = torch.LongTensor(convs).to(device)
        self.msk_conv = torch.LongTensor(conv_mask).to(device)
        self.msk_msg = torch.LongTensor(sent_mask).to(device)
        self.y = torch.LongTensor(target).to(device)


def text_generator(example_generator):
    for e in example_generator:
        id = e.features.feature['id'].bytes_list.value[0].decode('utf-8')
        conversation = e.features.feature['conv'].bytes_list.value[0].decode('utf-8')
        target = e.features.feature['target'].bytes_list.value[0].decode('utf-8')

        if len(conversation) == 0:
            continue

        yield id, conversation, target


def data_generator(input):
    filelist = glob.glob(input)
    assert filelist, ('Error: Empty filelist at %s' % input)

    filelist = sorted(filelist)

    for f in filelist:
        reader = open(f, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
