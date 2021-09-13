import sklearn.metrics as metric
import json
import torch
from transformers import InputExample
from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import numpy as np
import os
import pickle


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<unk>':0}
        self.word2count = {}
        self.index2word = {}
        self.word_count = 1

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.word_count
            self.index2word[self.word_count] = word
            self.word_count += 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)


def eval_to_file(out_file_name, preds, true_labels):
    accuracy = metric.accuracy_score(true_labels, preds)
    precision = metric.precision_score(true_labels, preds, average='macro')
    recall = metric.recall_score(true_labels, preds, average='macro')
    f1 = metric.f1_score(true_labels, preds, average='macro')
    MSE = metric.mean_squared_error(true_labels, preds)
    RMSE = metric.mean_squared_error(true_labels, preds, squared=False)

    model_statistics = {}
    with open(out_file_name, "w") as f:
        line = "accuracy: " + str(accuracy) + "\n" + "precision: " + str(precision) + "\n" + "recall: " + str(recall) + "\n" + "F1: " + str(f1) + "\n" + "MSE: " + str(MSE) + "\n" + "RMSE: " + str(RMSE) + "\n"
        model_statistics['statistics'] = line

        json.dump(model_statistics, f)


def remove_chars(text, target):
    for t in target:
        text = text.replace(t, "")
    return text


def load_data(args, data_dirs, tokenizer):
    up_vocab = Lang('user_product')
    datasets = []
    target = ['<sssss>']
    for dir in tqdm(data_dirs, desc="Loading dataset"):
        cache_dir = dir + ".cache_" + args.model_type
        if os.path.isfile(cache_dir):
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_user_ids, all_product_ids, up_vocab = pickle.load(open(cache_dir, 'rb'))
        else:
            examples = []
            user_ids, product_ids = [], []
            with open(dir, "r") as f:
                lines = f.readlines()
                lines = tqdm(lines, desc="lines")

                for i, line in enumerate(lines):
                    guid = "%s-%s" % ("document-sa", i)
                    a, b, c, d = line.split('\t\t')
                    text_a = remove_chars(d, target)

                    label = c
                    up_vocab.addWord(a)
                    up_vocab.addWord(b)

                    user_ids.append(up_vocab.word2index[a])
                    product_ids.append(up_vocab.word2index[b])
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
                    )
                features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    label_list=args.label_list,
                    max_length=args.max_seq_length,
                    output_mode=args.output_mode,
                    pad_on_left=False,
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                    pad_token_segment_id=0,
                )
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            if args.output_mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            if args.output_mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
            all_user_ids = torch.tensor(user_ids, dtype=torch.long)
            all_product_ids = torch.tensor(product_ids, dtype=torch.long)

            pickle.dump([all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_user_ids, all_product_ids, up_vocab], open(cache_dir, "wb"), protocol=4)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_user_ids, all_product_ids)

        datasets.append(dataset)
    datasets.append(up_vocab)
    return datasets


def get_label_distribution(datadir, reverse=True):
    sta = {}
    with open(datadir, " r") as f:
        lines = f.readlines()

        for line in lines:
            a, b, c, d = line.split('\t\t')
            label = int(c)
            if label not in sta:
                sta[label] = 0
            sta[label] += 1
        total = sum(sta[k] for k in sta)
        distri = [sta[e+1]/total for e in range(len(sta))]
        if reverse:
            distri = [1/e for e in distri]
        return distri
