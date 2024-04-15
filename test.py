from dataset.fewshot import FewShot
from model.CrackNex_matching import CrackNex
from util.utils import count_params, set_seed, mIOU

import argparse
import os
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Mining Latent Classes for Few-shot Segmentation')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='llCrackSeg9k',
                        choices=['llCrackSeg9k', 'LCSD'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')

    # few-shot training arguments
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--path',
                        type=str,
                        help='chekpoint path')
    parser.add_argument('--savepath',
                        type=str,
                        default='./logs/',
                        help='results saving path')
        
    args = parser.parse_args()
    return args


def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)
    
    num_classes = 3
    metric = mIOU(num_classes)

    for i, (img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_q, hiseq_q, mask_q = img_q.cuda(), hiseq_q.cuda(), mask_q.cuda()
        for k in range(len(img_s_list)):
            img_s_list[k], hiseq_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), hiseq_s_list[k].cuda(), mask_s_list[k].cuda()
        cls = cls[0].item()

        with torch.no_grad():
            out_ls = model(img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q)
            pred = torch.argmax(out_ls[0], dim=1)
        
        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        # if seed == 0:
        #     result = pred.squeeze(0).cpu().numpy().copy()
        #     result[result == 2] = 255
            
        #     im = Image.fromarray(np.uint8(result))
        #     im.save(args.savepath + id_q[0] + '.png')

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))
    
    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    save_path = 'outdir/models/%s' % (args.dataset)
    os.makedirs(save_path, exist_ok=True)

    testset = FewShot(args.data_root.replace('train_coco', 'val_coco'), None, 'val',
                      args.shot, 760 if args.dataset == 'LCSD' else 4000)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model = CrackNex(args.backbone)
    checkpoint_path = args.path

    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    #print(model)
    print('\nParams: %.1fM' % count_params(model))

    best_model = DataParallel(model).cuda()

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')


if __name__ == '__main__':
    main()

