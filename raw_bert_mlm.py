#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import paddle
import paddle.optimizer as opt
import paddle.distributed as dist
import paddle.fluid.dygraph as dygraph
from paddle.io import DistributedBatchSampler, IterableDataset, DataLoader
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.nn import TransformerEncoderLayer, TransformerEncoder
import paddle.nn as nn
from paddle.nn import Dropout
import paddle.nn.functional as F
import time
import random
import sys
import numpy as np
import pandas as pd
import re
import jieba
# import nltk
# nltk.download('punkt')
from datetime import datetime
from sklearn.metrics import roc_auc_score
import bz2
import zipfile
import json
import warnings
warnings.filterwarnings('ignore')
paddle.seed(2022)
random.seed(2022)

class TransformerEncoderLayer(nn.Layer):
    """ TransformerEncoderLayer"""
    def __init__(self, d_model, nhead, hidden_size):
        super(TransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.dim_head = d_model // nhead
        assert self.dim_head * nhead == d_model
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)
        self.linear_head = nn.Linear(d_model, d_model)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(0.1, mode="upscale_in_train")
        self.dropout2 = Dropout(0.1, mode="upscale_in_train")
        self.dropout3 = Dropout(0.1, mode="upscale_in_train")
        self.relu = nn.ReLU()

    def _process_qkv(self, x):
        x = paddle.reshape(x = x, shape = [0, 0, self.nhead, self.dim_head])
        x = paddle.transpose(x = x, perm = [0, 2, 1, 3])
        return x

    def MultiHeadAttention(self, input):
        q = self._process_qkv(self.wq(input))
        k = self._process_qkv(self.wk(input))
        v = self._process_qkv(self.wv(input))
        weight = F.softmax(paddle.matmul(q * (self.dim_head**0.5), paddle.transpose(x = k, perm = [0, 1, 3, 2])))
        val = paddle.matmul(weight, v)
        val = paddle.transpose(x = val, perm = [0, 2, 1, 3])
        val = paddle.reshape(x = val, shape = [0, 0, self.d_model])
        return val

    def ffn(self, input):
        input = self.dropout1(self.relu(self.linear1(input)))
        return self.linear2(input)

    def forward(self, input):
        input = self.layerNorm1(input + self.dropout3(self.MultiHeadAttention(input)))
        output = self.layerNorm2(input + self.dropout2(self.ffn(input)))
        return output


class bertModel(nn.Layer):
    """ bertModel """
    def __init__(self, d_model, nhead, n_layer, vocab_size, max_len, hidden_size):
        """ __init__ """
        super(bertModel, self).__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.encoder = nn.LayerList([TransformerEncoderLayer(d_model, nhead, hidden_size) for i in range(n_layer)])
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(self, input_token):
        """ forward """
        input_position = paddle.to_tensor([list(range(self.max_len))] * input_token.shape[0])
        enc_input = self.token_embedding(input_token) + self.position_embedding(input_position)
        for i, layer in enumerate(self.encoder):
            enc_input = layer(enc_input)
        output = self.dense(enc_input)
        return output

class DeBERTa_Dataset(IterableDataset):
    def __init__(self, input_file, tokenization_file, max_len):
        self.intpus = input_file
        self.tokenization = open(tokenization_file, 'r').read().split('\n')
        self.tokenization = dict(zip(self.tokenization, range(len(self.tokenization))))

        self.max_len = max_len

    def __process__(self, data):
        # tokens = [x for x in jieba.lcut(data.strip('\n')) if x not in ('')]
        tokens = list(data.strip('\n'))
        idx = [self.tokenization[x] for x in tokens if x in self.tokenization]
        MASK_ID = self.tokenization['[MASK]']
        PAD_ID = self.tokenization['[PAD]']
        valid_len = len(idx)
        rands = np.random.random(valid_len)
        mask_token, mask_idx = [], []
        for index, (r, w) in enumerate(zip(rands, idx)):
            if r < 0.15 * 0.8:
                mask_token.append(MASK_ID)
                mask_idx.append(1)
            elif r < 0.15 * 0.9:
                mask_token.append(w)
                mask_idx.append(1)
            elif r < 0.15:
                mask_token.append(np.random.choice(len(self.tokenization)))
                mask_idx.append(1)
            else:
                mask_token.append(w)
                mask_idx.append(0)
        if len(mask_token) > self.max_len:
            mask_token = mask_token[: self.max_len]
            mask_idx = mask_idx[: self.max_len]
            idx = idx[: self.max_len]
        else:
            mask_token += [PAD_ID] * (self.max_len - len(mask_token))
            mask_idx += [0] * (self.max_len - len(mask_idx))
            idx += [PAD_ID] * (self.max_len - len(idx))
        mask_token = paddle.to_tensor(mask_token, dtype = 'int32')
        mask_idx = paddle.to_tensor(mask_idx, dtype = 'int32')
        idx = paddle.to_tensor(idx, dtype = 'int32')
        return idx, mask_token, mask_idx

    def __iter__(self):
        azip = zipfile.ZipFile(self.intpus)
        self.cache = []
        self.cache_cnt = 10000
        for fi in azip.namelist():
            if fi.endswith('/'): continue
            for line in azip.open(fi):
                line = json.loads(line.decode("utf-8", "ignore"))
                text = line['title'].strip('\n') + ' ' + line['text'].strip('\n')
                self.cache.append(text)
                if len(self.cache) > self.cache_cnt:
                    text = random.choice(self.cache)
                    self.cache.remove(text)
                    idx, mask_token, mask_idx = self.__process__(text)
                    yield idx, mask_token, mask_idx
        while len(self.cache) > 0:
            text = random.choice(self.cache)
            self.cache.remove(text)
            idx, mask_token, mask_idx = self.__process__(text)
            yield idx, mask_token, mask_idx

def train_mlm(model, epochs, step_print, model_file, device, input_file, tokenization_file, max_len, batch_size):
    adam = opt.Adam(parameters=model.parameters(), learning_rate = 1.5e-4, weight_decay = 0.01)
    step = 0
    losses = 0
    ce_loss = paddle.nn.CrossEntropyLoss(reduction='none')
    for epoch in range(epochs):
        dataset = DeBERTa_Dataset(input_file, tokenization_file, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
        for idx, mask_token, mask_idx in loader:
            # print (mask_idx, idx)
            output = model(mask_token)
            idx = paddle.cast(idx, dtype='int64')
            loss = paddle.sum(ce_loss(output, idx) * mask_idx)
            adam.clear_grad()
            loss.backward()
            adam.step()
            model.clear_gradients()
            step += 1
            losses += loss.numpy().tolist()[0]
            if step % step_print == 0:
                print ('epoch: {}\tstep: {}\ttime: {}\tloss: {:.5f}'.format(epoch, step, datetime.now(), \
                    losses))
                losses = 0
                if device != 0:
                    break
        paddle.save(model.state_dict(), model_file + '_ecpoch_{}'.format(epoch))
    paddle.save(model.state_dict(), model_file)
    return model

def main(input_file, tokenization_file, model_file):
    try:
        device = paddle.distributed.ParallelEnv().dev_id
        print ('device: ', device, 'gpu')
        step_print = 100
        epochs = 100
        batch_size = 16
        nhead = 12
        n_layer = 12
        hidden_size = 1024
        d_model = 768
        max_len = 512
    except:
        step_print = 3
        epochs = 2
        batch_size = 4
        nhead = 2
        n_layer = 2
        hidden_size = 4
        d_model = 16
        max_len = 4
        device = 'cpu'
        print ('device cpu')
    vocab_size = len(open(tokenization_file, 'r').read().split('\n'))
    print ('vocab_size: {}'.format(vocab_size))
    model = bertModel(d_model, nhead, n_layer, vocab_size, max_len, hidden_size)
    model = train_mlm(model, epochs, step_print, model_file, device, input_file, tokenization_file, max_len, batch_size)


input_file = '../data/wiki_zh_2019.zip'
tokenization_file = '../data/token'
model_file = './output/model.pdparams'

# input_file = sys.argv[1]
# tokenization_file = sys.argv[2]
# model_file = sys.argv[3]
main(input_file, tokenization_file, model_file)