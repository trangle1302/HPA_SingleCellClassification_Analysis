from io import IncrementalNewlineDecoder
import sys
sys.path.insert(0, '..')
import argparse
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from importlib import import_module

import torch
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from utilities.augment_util import *
from utilities.model_util import load_pretrained
from dataset.cell_cls_dataset import CellClsDataset as HPA2021Dataset
from dataset.cell_cls_dataset import cls_collate_fn as collate_fn
cudnn.benchmark = True

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def initialize_environment(args):
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # COMMON
    args.device = 'cuda' if args.gpus else 'cpu'
    args.can_print = True
    args.seed = seed
    args.debug = False

    # MODEL
    args.scheduler = None
    args.loss = None
    args.pretrained = False

    # DATASET
    args.image_size = args.image_size.split(',')
    if len(args.image_size) == 1:
        args.image_size = [int(args.image_size[0]), int(args.image_size[0])]
    elif len(args.image_size) == 2:
        args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
    else:
        raise ValueError(','.join(args.image_size))

    args.num_workers = 4
    args.split_type = 'random'
    args.suffix = 'png'
    args.augments = args.augments.split(',')

    if args.can_print:
        if args.gpus:
            print(f'use gpus: {args.gpus}')
        else:
            print(f'use cpu')
    if args.gpus:
      os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.model_fpath = f'{DIR_CFGS.MODEL_DIR}/{args.model_dir}/fold{args.fold}/{args.model_epoch}.pth'
    args.output_dir = f'{DIR_CFGS.DATA_DIR}/1st_cams'
    args.feature_dir = f'{DIR_CFGS.FEATURE_DIR}/{args.model_dir}/fold{args.fold}/epoch_{args.model_epoch}'

    if args.can_print:
        print(f'output dir: {args.output_dir}')
        print(f'feature dir: {args.feature_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_dir, exist_ok=True)
    return args

def load_model(args):
    image_size = args.image_size
    args.image_size = image_size[0]
    model = import_module(f'net.{args.module}').get_model(args)[0]
    args.image_size = image_size
    model = load_pretrained(model, args.model_fpath, strict=True, can_print=args.can_print)
    model = model.eval().to(args.device)
    return model

def generate_dataloader(args):
    test_dataset = HPA2021Dataset(
      args,
      transform=None,
      dataset=args.dataset,
      is_training=False,
    )
    test_dataset.set_part(part_start=args.part_start, part_end=args.part_end)
    _ = test_dataset[0]
    test_loader = DataLoader(
      test_dataset,
      sampler=SequentialSampler(test_dataset),
      batch_size=1,
      drop_last=False,
      num_workers=args.num_workers,
      pin_memory=False,
      collate_fn=collate_fn,
    )
    print(f'num: {test_dataset.__len__()}')
    return test_loader

def generate_cam(args, test_loader, model):
    model = load_model(args) if model is None else model
    #
    print(model)
    print(model.fc_layers[2],model.fc_layers[4])
    print(model.backbone.layer4[-1])
    test_loader = generate_dataloader(args) if test_loader is None else test_loader
    target_layers = [model.backbone.layer4[-1]] #[model.fc_layers[-2]] #
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=('cuda' == args.device))
    
    augment='default' # default == nothing
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc=f'cell {augment}'):
        inp = iter_data['image'].to(args.device)
        rgb_img = np.array(iter_data['image'][0][0:3,:,:]*255).astype(np.uint8).transpose(1, 2, 0)
        # channels:['red', 'green', 'blue', 'yellow'] or MT, protein, nu, ER
        # cv2 is in format BGR
        bgr_img = rgb_img[...,[2,1,0]]
        cv2.imwrite(f"{args.output_dir}/{iter_data['ID'][0]}.png", bgr_img)
        for v,k in LABEL_TO_ALIAS.items():
            grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(v)], aug_smooth=True)
            visualization = show_cam_on_image(bgr_img/255, grayscale_cam[0], use_rgb=False)
            cv2.imwrite(f"{args.output_dir}/{iter_data['ID'][0]}_{iter_data['index'][0]}_{k}.png", visualization)

def main(args):
    start_time = timer()
    args = initialize_environment(args)
    model = None
    test_loader = None
    generate_cam(args, test_loader, model)
    end_time = timer()
    print(f'time: {(end_time - start_time) / 60.:.2f} min.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--module', type=str, default='cls_efficientnet', help='model')
    parser.add_argument('--model_name', type=str, default='cls_efficientnet_b3', help='model_name')
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir')
    parser.add_argument('--model_epoch', type=str, default='99.00_ema', help='model_epoch')
    parser.add_argument('--gpus', default=None, type=str, help='use gpus')
    parser.add_argument('--image_size', default='512', type=str, help='image_size')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--dataset', default='valid', type=str, help='dataset')
    parser.add_argument('--fold', default=0, type=int, help='index of fold')
    parser.add_argument('--augments', default='default', type=str, help='augments')
    parser.add_argument('--part_start', default=0., type=float, help='part_start')
    parser.add_argument('--part_end', default=1., type=float, help='part_end')
    parser.add_argument('--overwrite', default=0, type=int, help='overwrite')
    parser.add_argument('--ml_num_classes', default=ML_NUM_CLASSES, type=int)
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--num_classes', default=NUM_CLASSES, type=int)
    parser.add_argument('--cell_type', type=int, default=0)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--kaggle', type=int, default=0)
    parser.add_argument('--cell_complete', type=int, default=0)
    args = parser.parse_args()
    main(args)
