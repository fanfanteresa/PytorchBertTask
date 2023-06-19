# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertForSequenceClassification

import config
from data_loader import get_data_loader
from log import init_log
from utils.common_utils import set_seed, report
from utils.metric import calculate_metric


args = config.Args().get_parser()
set_seed(args.seed)

init_log(args.log_dir)
logger = logging.getLogger()

if args.use_tensorboard:
    writer = SummaryWriter(log_dir='./tensorboard')


class SentenceClassificationPipline:
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.label2id = {}
        idx = 0
        label_path = os.path.join(self.args.data_dir, "labels.txt")
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                ll = line.strip()
                self.label2id[ll] = idx
                idx += 1
        self.id2label = {}
        for key, value in self.label2id.items():
            self.id2label[value] = key

        self.softmax = nn.Softmax(-1)

        pretrained_path = args.bert_dir
        logging.info("Loading Bert pre-trained checkpoint: {}".format(pretrained_path))

        self.model = BertForSequenceClassification.from_pretrained(pretrained_path, num_labels=len(self.label2id),
                                                                   return_dict=False)

    def save_model(self, global_step):
        """根据global_step来保存模型"""
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir, exist_ok=True)
        output_model_file = os.path.join(self.args.save_dir, 'pytorch_model-ckpt-{}.bin'.format(global_step))
        logger.info('Saving model checkpoint to {}'.format(output_model_file))
        torch.save(self.model.state_dict(), output_model_file)

    def load_model(self):
        load_training_checkpoint = os.path.join(args.save_dir, "pytorch_model-ckpt-{}.bin".format(args.load_ckpt_id))
        checkpoint_state_dict = torch.load(load_training_checkpoint,
                                           map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint_state_dict)
        logger.info("Loading ckpt from {} succeed".format(load_training_checkpoint))

    def setup_optimizer(self):
        params = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        def __get_weight_decay_params():
            return [p for n, p in params if not any(nd in n for nd in no_decay)]

        def __get_no_decay_params():
            return [p for n, p in params if any(nd in n for nd in no_decay)]

        weight_decay_params = __get_weight_decay_params()
        no_decay_params = __get_no_decay_params()
        grouped_params = [
            {
                "params": weight_decay_params,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.lr
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": self.args.lr
            }
        ]
        optimizer = AdamW(grouped_params, correct_bias=False, eps=self.args.adam_epsilon)
        return optimizer

    def setup_scheduler(self, optimizer, t_total):
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

    def train(self):
        train_file_path = os.path.join(self.args.data_dir, "train.csv")
        train_loader = get_data_loader(train_file_path, self.label2id, self.args.bert_dir,
                                       self.args.max_seq_len, self.args.batch_size, True)
        t_total = len(train_loader) * self.args.train_epoch

        if self.args.dev:
            dev_file_path = os.path.join(self.args.data_dir, "dev.csv")
            dev_loader = get_data_loader(dev_file_path, self.label2id, self.args.bert_dir,
                                         self.args.max_seq_len, self.args.batch_size, False)

        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer, t_total)

        global_step = 0
        self.model.zero_grad()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        eval_step = self.args.eval_steps
        best_f1 = 0.
        for epoch in range(1, self.args.train_epoch + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(device)
                res = self.model(input_ids=batch_data['input_ids'],
                                 attention_mask=batch_data['attention_mask'],
                                 token_type_ids=batch_data['token_type_ids'],
                                 labels=batch_data['label_ids'])
                loss = res["loss"]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                logger.info('【train】 Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))

                if self.args.use_tensorboard == "True":
                    writer.add_scalar('data/loss', loss.item(), global_step)
                if global_step % eval_step == 0:
                    precision, recall, f1_score = self.dev(dev_loader)
                    logger.info(
                        '[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(precision, recall, f1_score))
                    if f1_score > best_f1:
                        self.save_model(global_step)
                        best_f1 = f1_score

    def dev(self, dev_loader):
        logits = []
        labels = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for eval_step, batch_data in enumerate(dev_loader):
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(device)
                res = self.model(input_ids=batch_data['input_ids'],
                                 attention_mask=batch_data['attention_mask'],
                                 token_type_ids=batch_data['token_type_ids'],
                                 labels=batch_data['label_ids'])
                logits.append(res["logits"])
                labels.append(batch_data['label_ids'])
        precision, recall, f1_score = calculate_metric(logits, labels)
        return precision, recall, f1_score

    def test(self, generate_report=False):
        test_file_path = os.path.join(self.args.data_dir, "test.csv")
        test_loader = get_data_loader(test_file_path, self.label2id, self.args.bert_dir,
                                      self.args.max_seq_len, self.args.batch_size, False)
        ori_data = test_loader.dataset.data

        self.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        probs = []
        labels = []
        with torch.no_grad():
            for test_step, test_batch_data in enumerate(test_loader):
                for key in test_batch_data.keys():
                    test_batch_data[key] = test_batch_data[key].to(device)
                res = self.model(input_ids=test_batch_data['input_ids'],
                                 attention_mask=test_batch_data['attention_mask'],
                                 token_type_ids=test_batch_data['token_type_ids'],
                                 labels=test_batch_data['label_ids'])
                probs.append(self.softmax(res["logits"]))
                labels.append(test_batch_data['label_ids'])

        precision, recall, f1_score = calculate_metric(probs, labels)
        logger.info("========test metric========")
        logger.info("precision:{} recall:{} f1:{}".format(precision, recall, f1_score))
        logger.info("========test metric========")

        if generate_report:
            report(args.report_dir, ori_data, probs, labels, self.id2label, True)
        return precision, recall, f1_score

    def predict(self, text, tokenizer, id2label):
        self.load_model()
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        with torch.no_grad():
            tokens = [i for i in text]
            if len(tokens) > self.args.max_seq_len - 2:
                tokens = tokens[:self.args.max_seq_len - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(token_ids)
            token_type_ids = [0] * len(token_ids)

            token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(device)
            attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.uint8)).unsqueeze(0).to(device)
            token_type_ids = torch.from_numpy(np.array(token_type_ids)).unsqueeze(0).to(device)
            res = self.model(input_ids=token_ids,
                             attention_mask=attention_masks,
                             token_type_ids=token_type_ids)

            probs = self.softmax(res["logits"])
            values, indices = probs.max(-1)
            prob = values.cpu().numpy().tolist()[0]
            pred_label_id = indices.cpu().numpy().tolist()[0]
            pred_label_name = id2label[pred_label_id]
        return prob, pred_label_id, pred_label_name

    def save_jit(self):
        test_file_path = os.path.join(self.args.data_dir, "test.csv")
        test_loader = get_data_loader(test_file_path, self.label2id, self.args.bert_dir,
                                      self.args.max_seq_len, self.args.batch_size, False)

        self.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for test_step, test_batch_data in enumerate(test_loader):
                for key in test_batch_data.keys():
                    test_batch_data[key] = test_batch_data[key].to(device)
                res = self.model(input_ids=test_batch_data['input_ids'],
                                 attention_mask=test_batch_data['attention_mask'],
                                 token_type_ids=test_batch_data['token_type_ids'])
                # import pdb;pdb.set_trace()
                traced_model = torch.jit.trace(self.model, (test_batch_data['input_ids'],
                                             test_batch_data['attention_mask'],
                                             test_batch_data['token_type_ids']))

                traced_model.save("saved_model.pt")
                break


if __name__ == "__main__":
    pipeline = SentenceClassificationPipline(args)
    pipeline.save_jit()
    # pipeline.train()
    # pipeline.test(False)






