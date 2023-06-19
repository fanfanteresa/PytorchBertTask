# -*- coding: utf-8 -*-
import logging
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger()


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """将序列padding到同一长度
    """
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    elif isinstance(inputs[0], torch.Tensor):
        assert mode == 'post', '"mode" argument must be "post" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


def set_seed(seed=123):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def report(output_dir, ori_data, probs, labels=None, id2label=None, output_error_file=False):
    """
    :param output_dir: 报告输出目录
    :param ori_data: 原始训练文字
    :param probs: 预测概率
    :param labels: 原始标签
    :param id2label: 标签id->标签名字的dict
    :param output_error_file: 预测错误的example是否单独生成文件
    :return:
    """
    y_pred = []
    y_true = []
    y_probs = []
    for p, l in zip(probs, labels):
        p = p.cpu()
        l = l.cpu()
        values, indices = p.max(-1)
        y_probs.extend(values.numpy().tolist())
        y_pred.extend(indices.numpy().tolist())
        y_true.extend(l.numpy().tolist())

    assert len(ori_data) == len(y_pred), \
        "ori_data size: {} does not match pred prob size: {}".format(len(ori_data), len(y_pred))

    buffer = []
    error_buffer = []
    for i in range(len(ori_data)):
        text = ori_data[i]
        pred_label = id2label[y_pred[i]]
        true_label = id2label[y_true[i]]
        pred_prob = y_probs[i]
        line = [element for element in text]
        line.extend([pred_label, pred_prob])
        buffer.append(line)
        if pred_label != true_label:
            error_buffer.append(line)

    model_report = pd.DataFrame(buffer)
    model_report.columns = ["text", "true_label", "pred_label", "prob"]
    error_report = pd.DataFrame(error_buffer)
    error_report.columns = ["text", "true_label", "pred_label", "prob"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if output_error_file:
        error_file = os.path.join(output_dir, "predict_error_report.csv")
        error_report.to_csv(error_file, index=False)
        logger.info("write error examples to {}".format(error_file))

    model_report_file = os.path.join(output_dir, "predict_report.csv")
    model_report.to_csv(model_report_file, index=False)
    logger.info("write predict examples to {}".format(model_report))









