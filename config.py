# -*- coding: utf-8 -*-
import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--save_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--load_ckpt_id', default=4,
                            help='the global step id for model checkpoints to load')

        parser.add_argument('--bert_dir', default='model_hub/chinese-bert-wwm/',
                            help='pretrained bert dir')
        parser.add_argument('--data_dir', default='data/',
                            help='input data dir')
        parser.add_argument('--log_dir', default='logs/')
        parser.add_argument('--report_dir', default='report/')


        # action
        parser.add_argument('--train', default=False, action='store_true')
        parser.add_argument('--dev', default=False, action='store_true')
        parser.add_argument('--test', default=False, action='store_true')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--max_seq_len', default=18, type=int)

        parser.add_argument('--eval_batch_size', default=2, type=int)

        # train args
        parser.add_argument('--train_epoch', default=15, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        # 2e-5
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='bert学习率')
        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')
        parser.add_argument('--use_tensorboard', default=False, action='store_true')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--eval_steps', default=2, type=int, help="多少步进行验证")

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
