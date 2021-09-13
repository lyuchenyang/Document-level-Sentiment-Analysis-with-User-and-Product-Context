""" running training and evaluation code for document-level sentiment analysis project

    Created by Chenyang Lyu
"""

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss

import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json

from os import listdir
from os.path import isfile, join

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertModel,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from modeling import SecondPretrainedBert, IncrementalContextBert, FocalLoss, IncrementalContextRoberta
from utils import eval_to_file, load_data
from pargs import Arguments

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
label_list = ["1", "2", "3", "4", "5"]
label_list_imdb = [str(i) for i in range(1, 11)]

MODEL_CLASSES = {
    "bert": (BertConfig, BertTokenizer, BertForSequenceClassification),
    "roberta": (RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification),
}

args = Arguments('document-level-sa')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def second_train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    for batch in tqdm(train_dataloader, desc="Second Train Iterating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "roberta":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )

            user, product = batch[4].view(-1, 1), batch[5].view(-1, 1)
            user_product = torch.cat([user, product], dim=1)
            _ = model(inputs, user_product)


def train(args, train_dataset, model, tokenizer, freeze=True, dev_set=None, eval_set=None, global_s=0):
    """ Training the model """
    tb_writer = SummaryWriter()

    num_labels = len(args.label_list)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) * args.num_train_epochs

    if args.is_second_training and freeze:
        model.embedding.weight.require_grad = False

    # Prepare optimizer for training
    if args.is_incremental and args.model_type == "roberta":
        large_lr = ["embedding.weight"]
        optimizer_group_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in large_lr)],
                "lr": args.learning_rate,
             },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in large_lr)],
                "lr": args.amply * args.learning_rate,
            }
        ]
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
             },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

    optimizer = AdamW(optimizer_group_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps*t_total), num_training_steps=t_total)

    # Train
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = global_s
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // len(train_dataloader // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    up_indices, new_embeddings = None, None

    loss_fct = CrossEntropyLoss()
    if args.is_focal_loss:
        loss_fct = FocalLoss(gamma=3, alpha=args.alpha)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            user, product = batch[4].view(-1, 1), batch[5].view(-1, 1)
            user_product = torch.cat([user, product], dim=1)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            if args.is_incremental:
                logits, up_indices, new_embeddings = model(inputs, user_product, up_indices, new_embeddings)
            else:
                inputs['labels'] = batch[3]
                outputs = model(**inputs)
                logits = outputs[1]  # model outputs are always tuple in transformers (see doc)

            loss = loss_fct(logits.view(-1, num_labels), batch[3].view(-1))

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if global_step % (5*args.logging_steps == 0) and dev_set is not None:
                        results = evaluate(args, model, dev_set, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)


    tb_writer.close()
    global_step = 1 if global_step ==0 else global_step

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.train_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        up_indices, new_embeddings = None, None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            user, product = batch[4].view(-1, 1), batch[5].view(-1, 1)
            user_product = torch.cat([user, product], dim=1)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                if args.model_type != 'roberta':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                if args.is_incremental:
                    logits, up_indices, new_embeddings = model(inputs, user_product)
                else:
                    inputs['labels'] = batch[3]
                    outputs = model(**inputs)
                    logits = outputs[1]  # model outputs are always tuple in transformers (see doc)
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = {'acc': (preds == out_label_ids).mean()}
        results.update(result)

        eval_to_file(args.eval_out_file, preds, out_label_ids)
        print("accuracy: ", metric.accuracy_score(out_label_ids, preds))
        print("precision: ", metric.precision_score(out_label_ids, preds))
        print("recall: ", metric.recall_score(out_label_ids, preds))
        print("F1: ", metric.f1_score(out_label_ids, preds))
        print("Mean Squared Error: ", metric.mean_squared_error(out_label_ids, preds))
        print("Root Mean Squared Error: ", metric.mean_squared_error(out_label_ids, preds, squared=False))

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, default="yelp-2013",
                        help="the name of the training task (the dataset name)")
    parser.add_argument("--model_size", type=str, default="base",
                        help="the size of pre-trained model")
    parser.add_argument("--model_type", type=str, default="bert",
                        help="the type of pre-trained model")
    parser.add_argument("--epochs", type=int, default=1,
                        help="the numebr of training epochs")
    parser.add_argument("--incremental", action="store_true",
                        help="use incremental mode")
    parser.add_argument("--second_train", action="store_true",
                        help="use second train mode")
    parser.add_argument("--do_train", action="store_true",
                        help="whether to train the model or not")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether to evaluate the model or not")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="the weight decay rate")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="the learning rate used to train the model")
    parser.add_argument("--warmup_steps", type=float, default=0.05,
                        help="the warm_up step rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="the maximum sequence length used to load dataset")
    parser.add_argument("--seed", type=int, default=1,
                        help="the random seed used in model initialization and dataloader")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="the batch size used in training and evaluation")
    parser.add_argument("--device", type=int, default=0,
                        help="the device id used for training and evaluation")
    parser.add_argument("--do_shrink", action="store_true",
                        help="whether to shrink the embedding size in order to reduce the amount of parameters")
    parser.add_argument("--inner_size", type=int, default=128,
                        help="the inner embedding size when do_shrink is true")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="the attention heads used in multi head attention function")
    parser.add_argument("--is_focal_loss", action="store_true",
                        help="whether to use focal loss function or not")

    arguments, _ = parser.parse_known_args()

    args.task_name = arguments.task_name
    args.model_size = arguments.model_size
    args.num_train_epochs = arguments.epochs
    args.is_incremental = arguments.incremental
    args.is_second_training = arguments.second_train
    args.do_train = arguments.do_train
    args.do_eval = arguments.do_eval
    args.weight_decay = arguments.weight_decay
    args.learning_rate = arguments.learning_rate
    args.warmup_steps = arguments.warmup_steps
    args.max_seq_length = arguments.max_seq_length
    args.seed = arguments.seed
    args.train_batch_size = arguments.batch_size
    args.device = torch.device("cuda:" + str(arguments.device))
    args.model_type = arguments.model_type
    args.is_focal_loss = arguments.is_focal_loss
    args.do_shrink = arguments.do_shrink

    if args.task_name == 'imdb':
        args.label_list = label_list_imdb
    else:
        args.label_list = label_list

    if args.is_incremental and args.is_second_training:
        raise ValueError("Incremental and second training modes can't be applied at the same time!")

    if args.model_type == 'bert':
        args.model_name_or_path = "bert-" + args.model_size + "-uncased"
    if args.model_type == 'roberta':
        args.model_name_or_path = "roberta-" + args.model_size

    if args.is_incremental:
        model_type = "incremental"
    else:
        model_type = 'vanilla'

    if args.is_second_training:
        model_type = 'second_train'

    if args.do_shrink:
        model_type += "_shrink"

    output_dir = "trained_model/" + args.model_name_or_path + "_" + args.task_name + '_' + model_type + "_epochs_" + str(args.num_train_epochs) + "_lr_" + \
        str(args.learning_rate) + "_weight-decay_" + str(args.weight_decay) + "_warmup_" + str(args.warmup_steps) + "_mql_" + str(args.max_seq_length) + '_shrink_' + str(arguments.inner_size) + \
        "_seed_" + str(args.seed) + "/"
    eval_dir = "eval_results/" + args.model_name_or_path + "_" + args.task_name + '_' + model_type + "_epochs_" + str(args.num_train_epochs) + "_lr_" + \
        str(args.learning_rate) + "_weight-decay_" + str(args.weight_decay) + "_warmup_" + str(args.warmup_steps) + "_mql_" + str(args.max_seq_length) + '_shrink_' + str(arguments.inner_size) + \
        "_seed_" + str(args.seed) + "/"

    args.output_dir = output_dir
    args.eval_out_file = eval_dir

    config_class, tokenizer_class, model_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
    )

    num_labels = len(args.label_list)

    data_dirs = ["data/document-level-sa-dataset/" + args.task_name + "-seg-20-20.train.ss",
                 "data/document-level-sa-dataset/" + args.task_name + "-seg-20-20.dev.ss",
                 "data/document-level-sa-dataset/" + args.task_name + "-seg-20-20.test.ss",
                 ]
    train_dataset, dev_dataset, test_dataset, up_vocab = load_data(args, data_dirs, tokenizer)

    # long_dev_dataset, long_test_dataset = load_dev_and_eval_data(args, data_dirs[1:], tokenizer, up_vocab)

    set_seed(args)
    args.model_type = args.model_type.lower()
    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    config.output_attention = True
    config.output_hidden_states = True
    config.inner_size = arguments.inner_size
    config.do_shrink = arguments.do_shrink
    config.attention_heads = arguments.attention_heads
    if args.is_incremental:
        model_class = IncrementalContextBert if args.model_type == "bert" else IncrementalContextRoberta
        model = model_class.from_pretrained(
            args.model_name_or_path,
            num_embeddings=up_vocab.word_count,
            up_vocab=up_vocab,
            config=config,
        )
    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, freeze=False, dev_set=dev_dataset, eval_set=test_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.is_second_training:
        st_model = SecondPretrainingBert.from_pretrained(
            args.model_name_or_path,
            num_embeddings=up_vocab.word_count,
            config=config
        )
        st_model.bert = model
        st_model.to(args.device)
        second_train(args, train_dataset, st_model)

        embedding_matrix = st_model.embedding_matrix

        new_model = IncrementalContextBert.from_pretrained(
            args.model_name_or_path,
            num_embeddings=up_vocab.word_count,
            up_vocab=up_vocab,
            config=config,
        )

        new_model.embedding = embedding_matrix

        args.is_incremental = True
        model = new_model
        model.to(args.device)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        # model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            if args.is_incremental or args.is_second_training:
                model = model_class.from_pretrained(
                    checkpoint,
                    num_embeddings=up_vocab.word_count,
                    up_vocab=up_vocab,
                    config=config,
                )
            else:
                model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, dev_dataset, tokenizer, prefix=prefix)
            result = evaluate(args, model, test_dataset, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == '__main__':
    main()
