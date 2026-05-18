import argparse
import time

from tqdm import tqdm

from engine.wrapper import ViLSSegWrapper

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.dataset import QaTa
import utils.config as config

from thop import profile


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

if __name__ == '__main__':

    args = get_parser()

    # load model
    model = ViLSSegWrapper(args)

    checkpoint = torch.load('xxx/medseg_max_dice.ckpt', map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint,strict=True)

    # dataloader
    ds_test = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer = pl.Trainer(accelerator='gpu',devices=1) 
    model.eval()

    with torch.no_grad():
        for data in tqdm(dl_test):
            with torch.autograd.profiler.profile() as prof:
                # 调用模型的前向传播函数，并传入输入数据
                model = model.cuda()
                data[0][0]=data[0][0].cuda()
                data[0][1]['input_ids']=data[0][1]['input_ids'].cuda()
                data[0][1]['attention_mask']=data[0][1]['attention_mask'].cuda()
                # torch.cuda.synchronize()
                start = time.time()
                out = model(data[0])
                # torch.cuda.synchronize()
                end = time.time()
                elapsed_time_ms = (end - start) * 1000
                print(elapsed_time_ms)
                flops, params = profile(model, inputs=(data[0],))
                print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2))) # 4111514624.0 25557032.0 res50

            break

    trainer.test(model, dl_test)

    print(trainer.logged_metrics)
