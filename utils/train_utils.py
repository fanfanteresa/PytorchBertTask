# -*- coding: utf-8 -*-
import logging
import os
import torch

logger = logging.getLogger()


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """
    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info('Load ckpt from {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info('Use multi gpus in: {}'.format(gpu_ids))
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info('Use single gpu in: {}'.format(gpu_ids))

    return model, device
