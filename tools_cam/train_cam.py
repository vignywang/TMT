
from numbers import Number
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import datetime
import pprint

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc
# from test_cam import val_loc_one_epoch
from test import val_loc_one_epoch_train

import json

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
import numpy as np
from re import compile


def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    print(cfg)
    model = create_deit_model(
        cfg.MODEL.ARCH,
        pretrained=True,
        num_classes=cfg.DATA.NUM_CLASSES,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])

            pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')
    optimizer = create_optimizer(args, model)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, cls_criterion


def main():
    args = update_config()
    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join(cfg.BASIC.SAVE_ROOT, 'ckpt', cfg.DATA.DATASET,
                                      '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
                                          cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE,
                                          cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log');
    mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt');
    mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, test_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = creat_model(cfg, args)

    best_gtknown = 0
    best_top1_loc = 0
    update_train_step = 0
    update_val_step = 0
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS + 1):

        adjust_learning_rate_normal(optimizer, epoch, cfg)
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion,
                            optimizer, epoch, writer, cfg, update_train_step)
        if epoch>=2:
           eval_results = val_loc_one_epoch_train(test_loader, model, device, )
           eval_results['epoch'] = epoch
           with open(os.path.join(cfg.BASIC.SAVE_DIR, 'val.txt'), 'a') as val_file:
               val_file.write(json.dumps(eval_results))
               val_file.write('\n')
           for k, v in eval_results.items():
               if isinstance(v, np.ndarray):
                   v = [round(out, 2) for out in v.tolist()]
               elif isinstance(v, Number):
                   v = round(v, 2)
               else:
                   raise ValueError(f'Unsupport metric type: {type(v)}')
               print(f'\n{k} : {v}')
           loc_gt_known = eval_results['GT-Known_top-1']

           if loc_gt_known > best_gtknown:
               best_gtknown = loc_gt_known
               torch.save({
                   "epoch": epoch,
                   'state_dict': model.state_dict(),
                   'best_map': best_gtknown
               }, os.path.join(ckpt_dir, f'model_best.pth'))

           print("Best GT_LOC: {}".format(best_gtknown))
           print("Best TOP1_LOC: {}".format(best_gtknown))


        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))



def train_one_epoch(train_loader, model, device, criterion, optimizer, epoch,
                    writer, cfg, update_train_step):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    log_var = ['module.layers.[0-9]+.fuse._loss_rate', 'module.layers.[0-9]+.thred']
    log_scopes = [compile(log_scope) for log_scope in log_var]

    model.train()
    for i, (input, target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)
        # cam = cam.to(device)
        vars = {}
        for log_scope in log_scopes:
            vars.update({key: val for key, val in model.named_parameters()
                         if log_scope.match(key)})

        # cls_logits = model(input, target ,return_cam=False)
        cls_logits, x_token= model(input, target, return_cam=False)
        loss1 = criterion(x_token, target)
        loss2 = criterion(cls_logits, target)
        loss = loss1 * 1 + loss2
        # loss =  loss2
        # loss = loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(cls_logits.data.contiguous(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('loss1_iter/train', loss1.item(), update_train_step)
        writer.add_scalar('loss2_iter/train', loss2.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top1', prec1.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top5', prec5.item(), update_train_step)

        for k, v in vars.items():
            writer.add_scalar(k, v.item(), update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader) - 1:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.7f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.7f})\t'
                   'Loss1 {loss1.val:.4f} ({loss1.avg:.7f})\t'
                   'Loss2 {loss2.val:.4f} ({loss2.avg:.7f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), loss=losses, loss1=losses1, loss2=losses2,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    return update_train_step, losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
