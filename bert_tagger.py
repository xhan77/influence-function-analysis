from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle

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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare data processor
    mnli_processor = MnliProcessor()
    hans_processor = HansProcessor()
    sst_processor = Sst2Processor()
    if args.task == "SA":
        label_list = sst_processor.get_labels()
    elif args.task == "NLI":
        label_list = mnli_processor.get_labels()
    else:
        raise ValueError("")
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare training data
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        if args.task == "SA":
            train_examples = sst_processor.get_train_examples(args.data_dir, args.num_train_samples)
        elif args.task == "NLI":
            train_examples = mnli_processor.get_train_examples(args.data_dir, args.num_train_samples)
        
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size) * args.num_train_epochs

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
    if args.trained_model_dir: # load in fine-tuned (with cloze-style LM objective) model
        if os.path.exists(os.path.join(args.output_dir, WEIGHTS_NAME)):
            previous_state_dict = torch.load(os.path.join(args.output_dir, WEIGHTS_NAME))
        else:
            from collections import OrderedDict
            previous_state_dict = OrderedDict()
        distant_state_dict = torch.load(os.path.join(args.trained_model_dir, WEIGHTS_NAME))
        previous_state_dict.update(distant_state_dict) # note that the final layers of previous model and distant model must have different attribute names!
        model = MyBertForSequenceClassification.from_pretrained(args.trained_model_dir, state_dict=previous_state_dict, num_labels=num_labels)
    else:
        model = MyBertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    if args.freeze_bert: # freeze BERT if needed
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
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(fr in n for fr in frozen)) and (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if (not any(fr in n for fr in frozen)) and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
        ]
    if args.fp16:
        raise ValueError("Not sure if FP16 precision works yet.")
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.do_train:
        global_step = 0
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
#         model.eval() # train in eval mode to avoid dropout
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_loss = []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                        
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_loss.append(loss.item())
            logger.info("  epoch loss = %f", np.mean(epoch_loss))

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_test:
        if args.task == "SA":
            test_examples = sst_processor.get_dev_examples(args.data_dir)
        elif args.task == "NLI":
            test_examples = mnli_processor.get_dev_examples(args.data_dir)

        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running final test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_guid = torch.tensor([f.guid for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guid)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        wrong_list = []

        for input_ids, input_mask, segment_ids, label_ids, guids in tqdm(test_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_test_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            tmp_test_correct, tmp_test_total = accuracy(logits, label_ids)
            
            assert tmp_test_total == 1
            if tmp_test_correct == 0:
                wrong_list.append(guids[0].item())

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_correct

            nb_test_examples += tmp_test_total
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples
        result = {'test_loss': test_loss,
                  'test_accuracy': test_accuracy}

        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                
#         pickle.dump(wrong_list, open(os.path.join(args.output_dir, "wrong_pred_guid.txt"), "wb"))

if __name__ == "__main__":
    main()
