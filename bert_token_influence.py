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

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

from bert_util import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True,
                        help="Sentiment analysis or natural language inference? (SA or NLI)")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned (with the cloze-style LM objective) BERT model?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Whether to freeze BERT")
    parser.add_argument('--full_bert',
                        action='store_true',
                        help="Whether to use full BERT")
    parser.add_argument('--num_train_samples',
                        type=int,
                        default=-1,
                        help="-1 for full train set, otherwise please specify")
    parser.add_argument('--damping',
                        type=float,
                        default=0.0,
                        help="probably need damping for deep models")
    parser.add_argument('--test_idx',
                        type=int,
                        default=1,
                        help="test index we want to examine")
    parser.add_argument('--influence_on_decision',
                        action='store_true',
                        help="Whether to compute influence on decision (rather than influence on ground truth)")
    parser.add_argument("--if_compute_saliency",
                        default=1,
                        type=int)
    parser.add_argument('--start_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--end_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument("--lissa_repeat",
                        default=1,
                        type=int)
    parser.add_argument("--lissa_depth",
                        default=1.0,
                        type=float)
    parser.add_argument('--mask_token',
                        action='store_true',
                        help="mask token and compute influence")
    parser.add_argument('--wrt_token',
                        action='store_true',
                        help="compute influence w.r.t. token saliency")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.influence_on_decision:
        raise ValueError("To use loss function w.r.t. the ground truth, manually disable this error in the code.")
    if args.if_compute_saliency == 0:
        raise ValueError("Must compute saliency for token level influence.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        logger.info("WARNING: Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mnli_processor = MnliProcessor()
    hans_processor = HansProcessor()
    sst_processor = Sst2Processor()
    if args.task == "SA":
        label_list = sst_processor.get_labels()
    elif args.task == "NLI":
        label_list = mnli_processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    model = MyBertForSequenceClassification.from_pretrained(args.trained_model_dir, num_labels=num_labels)
    if args.fp16:
        raise ValueError("Not sure if FP16 precision works yet.")
        model.half()
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
#     for n, p in param_optimizer:
#         print(n)
#     sys.exit()
    if args.freeze_bert:
        frozen = ['bert']
    elif args.full_bert:
        frozen = []
    else:
        frozen = ['bert.embeddings.',
                  'bert.encoder.layer.0.',
                  'bert.encoder.layer.1.',
                  'bert.encoder.layer.2.',
                  'bert.encoder.layer.3.',
                  'bert.encoder.layer.4.',
                  'bert.encoder.layer.5.',
                  'bert.encoder.layer.6.',
                  'bert.encoder.layer.7.',
                 ] # *** change here to filter out params we don't want to track ***

    param_influence = []
    for n, p in param_optimizer:
        if (not any(fr in n for fr in frozen)):
            param_influence.append(p)
        elif 'bert.embeddings.word_embeddings.' in n:
            pass # need gradients through embedding layer for computing saliency map
        else:
            p.requires_grad = False
            
    param_shape_tensor = []
    param_size = 0
    for p in param_influence:
        tmp_p = p.clone().detach()
        param_shape_tensor.append(tmp_p)
        param_size += torch.numel(tmp_p)
    logger.info("  Parameter size = %d", param_size)
    
    if args.task == "SA":
        train_examples = sst_processor.get_train_examples(args.data_dir, args.num_train_samples)
    elif args.task == "NLI":
        train_examples = mnli_processor.get_train_examples(args.data_dir, args.num_train_samples)

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Train set *****")
    logger.info("  Num examples = %d", len(train_examples))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guids)
    train_dataloader_wbatch = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=args.train_batch_size)
    train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1)
    
    if args.task == "SA":
        test_examples = sst_processor.get_dev_examples(args.data_dir)
    elif args.task == "NLI":
        test_examples = hans_processor.get_test_examples(args.data_dir)
    
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Test set *****")
    logger.info("  Num examples = %d", len(test_examples))
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guids)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=1)
    
    damping = args.damping
    
    test_idx = args.test_idx
    start_test_idx = args.start_test_idx
    end_test_idx = args.end_test_idx
    
    for input_ids, input_mask, segment_ids, label_ids, guids in test_dataloader:
        model.eval()
        
        guid = guids[0].item() # test set loader must have a batch size of 1 now
        if start_test_idx != -1 and end_test_idx != -1:
            if guid < start_test_idx:
                continue
            if guid > end_test_idx:
                break
        else:
            if guid < test_idx:
                continue
            if guid > test_idx:
                break
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        ######## GET TEST EXAMPLE DECISION ########
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)
            pred_label_ids = torch.from_numpy(outputs).long().to(device)
            if label_ids.item() == pred_label_ids.item():
                test_pred_status = "correct"
            else:
                test_pred_status = "wrong"
        if args.influence_on_decision:
            label_ids = torch.from_numpy(outputs).long().to(device)
        ################
        
        ######## TEST EXAMPLE SALIENCY MAP ########
        if args.if_compute_saliency:
            saliency_scores = saliency_map(model, input_ids, segment_ids, input_mask, pred_label_ids)
            test_tok_sal_list = []
            for tok, sal in zip(tokenizer.convert_ids_to_tokens(input_ids.view(-1).cpu().numpy()), saliency_scores):
                if tok == '[PAD]':
                    break
                test_tok_sal_list.append((tok, sal))
            pickle.dump((test_tok_sal_list, [], test_pred_status), open(os.path.join(args.output_dir, "saliency_test_" + str(guid) + ".pkl"), "wb"))
        ################
            
        ######## COMPUTE INFLUENCE WITH TOKENS MASKED ########
        if args.mask_token:
            diff_input_masks, mask_ix = get_diff_input_masks(input_mask, test_tok_sal_list)
            diff_influences_list = []
            for tok_i, diff_input_mask in enumerate(diff_input_masks):
                model.eval()
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                train_dataloader_lissa = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
                
                logger.info("computing influence with token masked: " + str(tok_i))
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, diff_input_mask)
                    logits = logits.detach().cpu().numpy()
                    outputs = np.argmax(logits, axis=1)
                if args.influence_on_decision:
                    label_ids = torch.from_numpy(outputs).long().to(device)

                model.zero_grad()
                test_loss = model(input_ids, segment_ids, diff_input_mask, label_ids)
                test_grads = autograd.grad(test_loss, param_influence)

                model.train()
                inverse_hvp = get_inverse_hvp_lissa(test_grads, model, device, param_influence, train_dataloader_lissa, damping=damping, num_samples=args.lissa_repeat, recursion_depth=int(len(train_examples)*args.lissa_depth))

                diff_influences = np.zeros(len(train_dataloader.dataset))
                for train_idx, (_input_ids, _input_mask, _segment_ids, _label_ids, _) in enumerate(tqdm(train_dataloader, desc="Train set index")):
                    model.train()
                    _input_ids = _input_ids.to(device)
                    _input_mask = _input_mask.to(device)
                    _segment_ids = _segment_ids.to(device)
                    _label_ids = _label_ids.to(device)
                    
                    model.zero_grad()
                    train_loss = model(_input_ids, _segment_ids, _input_mask, _label_ids)
                    train_grads = autograd.grad(train_loss, param_influence)
                    diff_influences[train_idx] = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()
                diff_influences_list.append(diff_influences)
            
            if args.influence_on_decision:
                pickle.dump((diff_influences_list, mask_ix), open(os.path.join(args.output_dir, "diff_mask_influences_test_" + str(guid) + ".pkl"), "wb"))
            else:
                pickle.dump((diff_influences_list, mask_ix), open(os.path.join(args.output_dir, "diff_mask_influences_on_x_test_" + str(guid) + ".pkl"), "wb"))
        ################


if __name__ == "__main__":
    main()
