'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import os
import numpy as np
import random
import time
import datetime
import json
import torch
import utils as utils
import torch.backends.cudnn as cudnn

from pathlib import Path
from models.blip import blip_decoder
from utils import cosine_lr_schedule
from data import create_dataset
from data.utils import coco_caption_eval


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        loss = model(image, caption)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()
    result = {}
    ground_true = {}
    for image, gold_captions, image_id in data_loader:
        
        image = image.to(device)
        pred_captions = model.generate(image, sample=True, max_length=config['max_length'],
                                       min_length=config['min_length'])
        
        for pred_caption, gold_caption, img_id in zip(pred_captions, gold_captions, image_id):
            result[str(img_id)] = [pred_caption]
            ground_true[str(img_id)] = [gold_caption]
    
    return result, ground_true


def main(config):
    device = torch.device(config['device'])
    
    # fix the seed for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating captioning dataset")
    train_loader, val_loader = create_dataset(config)
    
    #### Model ####
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    # model.load_state_dict(torch.load('output/checkpoint_best.pth')['model'])
    
    #### Train ####
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()
    train_stats = {}
    for epoch in range(0, config['max_epoch']):
        if not config['evaluate']:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device)
        
        val_result, val_ground_true = evaluate(model, val_loader, device, config)
        coco_val = coco_caption_eval(val_ground_true, val_result)
        
        if coco_val['rouge'] + coco_val['bleu3'] > best:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            
            best = coco_val['rouge'] + coco_val['bleu3']
            best_epoch = epoch
            torch.save(save_obj, os.path.join(config['output_dir'], 'checkpoint_best.pth'))
        
        print(json.dumps(coco_val, indent=4))
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in coco_val.items()},
                     'epoch': epoch,
                     'best_epoch': best_epoch,
                     }
        
        open(os.path.join(config['output_dir'], "log.txt"), "a").write(json.dumps(log_stats) + "\n")
        open(os.path.join(config['output_dir'], "result.txt"), "a").write(json.dumps(
            {f'epoch_{epoch}': {'val_result': val_result, 'val_ground_true': val_ground_true}}
        ) + "\n")
        
        if config['evaluate']: break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"The Best score of rouge+bleu3 is {best} of epoch {best_epoch}")
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    config = {
        # set pretrained as a file path or an url
        'pretrained': 'checkpoint/model_base_caption_capfilt_large.pth',
        # size of vit model; base or large
        'vit': 'base',
        'vit_grad_ckpt': False,
        'vit_ckpt_layer': 0,
        'batch_size': 8,
        'init_lr': 1e-5,
        'image_size': 384,
        # generation configs
        'max_length': 50,
        'min_length': 5,
        'prompt': '',
        # optimizer
        'weight_decay': 0.02,
        'min_lr': 0,
        'max_epoch': 50,
        # args
        'eval_ratio': 0.3,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'distributed': False,
        'evaluate': False,
        'output_dir': 'output/',
    }
    
    print(json.dumps(config, indent=4))
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    main(config)
