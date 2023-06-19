# -*- coding: utf-8 -*-
import torch
from torch import nn


class SoftmaxNN(nn.Module):
    def __init__(self, sentence_encoder, num_class):
        super(SoftmaxNN, self).__init__()
        self.base_model_prefix = "bert"

        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.bert.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(p=0.1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        """
        前向传播
        :param args: depends on the encoder
        :return: logits, (B, N)
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        labels = None
        if "labels" in inputs:
            labels = inputs["inputs"]

        rep = self.sentence_encoder(input_ids, attention_mask, token_type_ids)  # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep)  # (B, N)
        probs = self.softmax(logits)
        res = {
            "logits": logits,
            "probs": probs
        }
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            res["loss"] = loss
        return res


if __name__ == "__main__":
    from encoder.bert_encoder import BERTEncoder

    input_ids = torch.tensor([[101, 3118, 802, 2140, 118, 674, 6809, 2408, 1767, 102, 0, 0,
                               0, 0, 0, 0, 0, 0],
                              [101, 6568, 802, 6858, 118, 6568, 802, 6858, 118, 1922, 753, 7000,
                               5831, 7824, 102, 0, 0, 0]], dtype=torch.long, device='cpu')
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                  dtype=torch.long, device='cpu')
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
                                  dtype=torch.long, device='cpu')
    label_ids = torch.tensor([4, 2])
    dummy_inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": label_ids
    }

    bert_encoder = BERTEncoder("/Users/chenliu/Desktop/projects/LV55_INC_OUT_CLS/model_hub/chinese-bert-wwm/")
    # x = bert_encoder(input_ids, attention_mask, token_type_ids)
    classification_model = SoftmaxNN(bert_encoder, 84)
    # res = classification_model(input_ids=input_ids, attention_mask=attention_mask,
    #                                     token_type_ids=token_type_ids, label_ids=label_ids)

    res = classification_model(**dummy_inputs)
    print(res["logits"])
    print(res["loss"])








