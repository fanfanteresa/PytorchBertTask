# -*- coding: utf-8 -*-
import logging
from typing import List

import torch
import torch.utils.data as data
from transformers import BertTokenizer

from utils.common_utils import sequence_padding

logger: logging.Logger = logging.getLogger()


class ListDataset(data.Dataset):
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path


class TextClassificationDataset(ListDataset):
    """Sentence-level classification dataset."""

    @staticmethod
    def load_data(filename) -> List[list]:
        examples = []
        with open(filename, encoding="utf-8") as f:
            for line in f.readlines():
                elements = line.strip().split("\t")
                if len(elements) != 2:
                    logger.error("wrong input:{}".format(line))
                    continue
                text = elements[0]
                label = elements[1]
                examples.append([text, label])
        return examples

    @staticmethod
    def load_label(label_path):
        label2id = {}
        idx = 0
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                ll = line.strip()
                label2id[ll] = idx
                idx += 1

        return label2id


class TextClassificationIterableDataset(data.IterableDataset):
    def __init__(self, file_path):
        super(TextClassificationIterableDataset).__init__()
        self.data = self.load_data(file_path)

    def __iter__(self):
        return iter(self.data)

    @staticmethod
    def load_data(filename) -> List[list]:
        examples = []
        with open(filename, encoding="utf-8") as f:
            for line in f.readlines():
                elements = line.strip().split("\t")
                if len(elements) != 2:
                    logger.error("wrong input:{}".format(line))
                    continue
                text = elements[0]
                label = elements[1]
                examples.append([text, label])
        return examples


class Collate:
    def __init__(self, max_len, label2id, device, tokenizer):
        self.maxlen = max_len
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.device = device
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        batch_token_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        for _, (text, text_label) in enumerate(batch):
            if len(text) > self.maxlen - 2:
                text = text[:self.maxlen - 2]
            tokens = [i for i in text]
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            batch_token_ids.append(token_ids)  # 前面已经限制了长度
            batch_attention_mask.append([1] * len(token_ids))
            batch_token_type_ids.append([0] * len(token_ids))
            batch_labels.append(self.label2id[text_label])

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=self.maxlen), dtype=torch.long,
                                       device=self.device)
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=self.maxlen),
                                            dtype=torch.long,
                                            device=self.device)
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=self.maxlen),
                                            dtype=torch.long,
                                            device=self.device)
        batch_label_ids = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
        res = {
            "input_ids": batch_token_ids,
            "token_type_ids": batch_token_type_ids,
            "attention_mask": batch_attention_mask,
            "label_ids": batch_label_ids,  # (B)
        }
        return res


def get_data_loader(file_path, label2id, vocab_path, max_len, batch_size, shuffle=True):
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    dataset = TextClassificationDataset(file_path=file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = Collate(max_len=max_len, label2id=label2id, device=device, tokenizer=tokenizer)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                  collate_fn=collate.collate_fn)
    return data_loader


if __name__ == "__main__":
    max_len = 18
    tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm/')
    train_dataset = TextClassificationDataset(file_path='data/train.csv', label_path="data/labels.txt",
                                              tokenizer=tokenizer, max_len=max_len)
    print(train_dataset[0])

    label_path = "data/labels.txt"
    label2id = TextClassificationDataset.load_label(label_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = Collate(max_len=max_len, label2id=label2id, device=device, tokenizer=tokenizer)
    # res = collate.collate_fn(train_dataset[:4])
    # print(res)

    batch_size = 2
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                       collate_fn=collate.collate_fn)

    for _, batch in enumerate(train_dataloader):
        # print(f"input_ids.shape: {batch['input_ids'].shape}")
        # print(f"token_type_ids.shape: {batch['token_type_ids'].shape}")
        # print(f"attention_mask.shape: {batch['attention_mask'].shape}")
        # print(f"label_ids.shape: {batch['label_ids'].shape}")
        print(f"input_ids: {batch['input_ids']}")
        print(f"token_type_ids: {batch['token_type_ids']}")
        print(f"attention_mask: {batch['attention_mask']}")
        print(f"label_ids: {batch['label_ids']}")
