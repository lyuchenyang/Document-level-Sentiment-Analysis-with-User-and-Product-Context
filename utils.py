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

def convert_examples_to_features(examples, tokenizer, max_length=512, label_list=None, output_mode=None, pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    over_len = 0
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        encoded_text = tokenizer.encode(example.text_a)
        if len(encoded_text) > max_length:
            over_len += 1
            input_ids = encoded_text[:129] + encoded_text[-383:]
            token_type_ids = [0] * max_length
        else:
            inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    print(float(over_len/len(examples)))
    return features


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
