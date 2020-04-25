from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import torch.autograd as autograd
from scipy import stats

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class MyLSTMForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyLSTMForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.my_word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        input_ids_lengths = (input_ids > 0).sum(dim=1)
        words_embeddings = self.my_word_embeddings(input_ids)
        
        packseq = nn.utils.rnn.pack_padded_sequence(words_embeddings, input_ids_lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)
        
#         last_hidden = torch.cat([h[0], h[1]], dim=-1)

        logits = self.classifier(h[0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note=""):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid

class MnliProcessor(object):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, num_train_samples=-1):
        """See base class."""
        if num_train_samples != -1:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "mnli_train.tsv")), "mnli_train")[: num_train_samples]
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mnli_train.tsv")), "mnli_train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mnli_dev.tsv")), "mnli_dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "non-entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if label == "contradiction" or label == "neutral":
                label = "non-entailment" # collapse contradiction into non-entailment
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class HansProcessor(object):

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "small_heuristics_evaluation_set.txt")), "HANS small")
    
    def get_neg_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "negated_small_heuristics_evaluation_set.txt")), "HANS small negated")

    def get_labels(self):
        """See base class."""
        return ["entailment", "non-entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[5]
            text_b = line[6]
            label = line[0]
            note = line[8]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, note=note))
        return examples

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
class Sst2Processor(object):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, num_train_samples=-1):
        """See base class."""
        if num_train_samples != -1:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst2_train.tsv")), "train")[: num_train_samples]
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sst2_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sst2_dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, label_ids):
    # axis-0: seqs in batch; axis-1: potential labels of seq
    outputs = np.argmax(out, axis=1)
    matched = outputs == label_ids
    num_correct = np.sum(matched)
    num_total = len(label_ids)
    return num_correct, num_total



################ functions for influence function ################

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def unflatten_to_param_dim(x, param_shape_tensor):
    tar_p = []
    ptr = 0
    for p in param_shape_tensor:
        len_p = torch.numel(p)
        tmp = x[ptr : ptr + len_p].view(p.shape)
        tar_p.append(tmp)
        ptr += len_p
    return tar_p

def hv(loss, model_params, v): # according to pytorch issue #24004
#     s = time.time()
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
#     e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
#     e2 = time.time()
#     print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
    return Hv

######## LiSSA ########

def get_inverse_hvp_lissa(v, model, device, param_influence, train_loader, damping, num_samples, recursion_depth, scale=1e4):
    ihvp = None
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in range(recursion_depth):
            try:
                input_ids, input_mask, segment_ids, label_ids, guids = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                input_ids, input_mask, segment_ids, label_ids, guids = next(lissa_data_iterator)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            model.zero_grad()
            train_loss = model(input_ids, segment_ids, input_mask, label_ids)
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            if (j % 200 == 0) or (j == recursion_depth - 1):
                print("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    return return_ihvp

################

# adapted from AllenNLP Interpret
def _register_embedding_list_hook(model, embeddings_list, model_type):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    if model_type == 'BERT':
        embedding_layer = model.bert.embeddings.word_embeddings
    elif model_type == 'LSTM':
        embedding_layer = model.my_word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def _register_embedding_gradient_hooks(model, embeddings_gradients, model_type):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])
    if model_type == 'BERT':
        embedding_layer = model.bert.embeddings.word_embeddings
    elif model_type == 'LSTM':
        embedding_layer = model.my_word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency_map(model, input_ids, segment_ids, input_mask, pred_label_ids, model_type='BERT'):
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list, model_type)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients, model_type)

    model.zero_grad()
    _loss = model(input_ids, segment_ids, input_mask, pred_label_ids)
    _loss.backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    saliency_grad = np.sum(saliency_grad[0] * embeddings_list[0], axis=1)
    norm = np.linalg.norm(saliency_grad, ord=1)
#     saliency_grad = [math.fabs(e) / norm for e in saliency_grad]
    saliency_grad = [(- e) / norm for e in saliency_grad] # negative gradient for loss means positive influence on decision
    return saliency_grad

################

def get_diff_input_masks(input_mask, test_tok_sal_list):
    sal_scores = np.array([sal for tok, sal in test_tok_sal_list])
    sal_ordered_ix = np.argsort(sal_scores)
    invalid_ix = []
    for i, (tok, sal) in enumerate(test_tok_sal_list):
        if tok == '[CLS]' or tok == '[SEP]' or '##' in tok: # would not mask [CLS] or [SEP]
            invalid_ix.append(i)
    cleaned_sal_ordered_ix = []
    for sal_ix in sal_ordered_ix:
        if sal_ix in invalid_ix:
            continue
        else:
            cleaned_sal_ordered_ix.append(sal_ix)
            
    # add zero and random
    abs_sal_ordered_ix = np.argsort(np.absolute(sal_scores))
    cleaned_abs_sal_ordered_ix = []
    for sal_ix in abs_sal_ordered_ix:
        if sal_ix in invalid_ix:
            continue
        else:
            cleaned_abs_sal_ordered_ix.append(sal_ix)
    
#     mask_ix = (cleaned_sal_ordered_ix[0], cleaned_sal_ordered_ix[int(len(cleaned_sal_ordered_ix)/2)], cleaned_sal_ordered_ix[-1])
    mask_ix = (cleaned_sal_ordered_ix[0], cleaned_sal_ordered_ix[int(len(cleaned_sal_ordered_ix)/2)], cleaned_sal_ordered_ix[-1], cleaned_abs_sal_ordered_ix[0], random.choice(cleaned_sal_ordered_ix)) # lowest, median, highest, zero, random
    diff_input_masks = []
    for mi in mask_ix:
        diff_input_mask = input_mask.clone()
        diff_input_mask[0][mi] = 0
        diff_input_masks.append(diff_input_mask)
    return diff_input_masks, mask_ix

def influence_distance(orig_influences, alt_influences, top_percentage=0.01):
    orig_influences = stats.zscore(orig_influences)
    alt_influences = stats.zscore(alt_influences)
    orig_sorted_ix = list(np.argsort(orig_influences))
    orig_sorted_ix.reverse()
    alt_sorted_ix = list(np.argsort(alt_influences))
    alt_sorted_ix.reverse()
    num_top = int(len(orig_influences) * top_percentage)
    
    orig_top_ix = orig_sorted_ix[:num_top]
    alt_top_ix = alt_sorted_ix[:num_top]
    orig_top_ix_set = set(orig_top_ix)
    alt_top_ix_set = set(alt_top_ix)
    ix_intersection = list(orig_top_ix_set.intersection(alt_top_ix_set))
    
    return len(ix_intersection) / num_top
