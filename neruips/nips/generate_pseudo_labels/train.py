import torch
import numpy as np
import os, argparse
import torch.nn.functional as F

from model import CoattentionModel
from datetime import datetime
from dataset import PairwiseImg
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
parser.add_argument('--data_root', type=str, default=' ', help='video dataroot')
parser.add_argument('--img_path', type=str, default=' ', help='static-img dataroot')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--start-epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
parser.add_argument('--save_path', type=str, default='.\\save_output\\',
                    help='The directory used to save the trained models')

parser.add_argument('--mode', type=str, default='train', help='train mode ')
parser.add_argument('--epochs', type=int, default=70, help='number of total epochs to run')
parser.add_argument('--restore_from', type=str, default=' ',
                    help='the pretrained resnet-50 on DUTS')

parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--output_stride', type=int, default=16)

args = parser.parse_args()

def ownMinMax(x):
    B,C,H,W = x.shape
    norm_list = []
    for i in range(B):
        img = x[i,0,:,:]
        min = torch.min(img)
        max = torch.max(img)
        dis = max - min
        norm_img = (img - min) / dis
        norm_img = 1 - norm_img
        norm_img = torch.unsqueeze(norm_img, 0)
        norm_list.append(norm_img)
    norm_tensor = torch.cat(norm_list, 0)
    norm_tensor = torch.unsqueeze(norm_tensor, 1)
    norm_tensor = norm_tensor.detach()
    return norm_tensor

def make_weight(pre, label):
    pre = pre.detach()
    wbce_per = F.binary_cross_entropy_with_logits(pre, label, reduce=False)
    wbce_per = ownMinMax(wbce_per)
    wbce_per = 1 - wbce_per
    weight = torch.where(wbce_per > 0.03, 1, 0)
    return weight.float()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def weight_loss(pred, mask, weight):  # 两个都是32*1*352*352

    weit = 1 + 5 * weight
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main():
    model = CoattentionModel(num_classes=1, all_channel=256, all_dim=56 * 56)
    print('Load pretrained ', args.restore_from, '...')
    pretrain_dict = torch.load(args.restore_from)
    resnet_part_params = {}
    state_dict = model.state_dict()

    for k,v in state_dict.items():
        if 'module.' + k[8:] in pretrain_dict.keys():
            v = pretrain_dict['module.' + k[8:]]
            resnet_part_params[k] = v
        else:
            print('missing keys :', k)
    state_dict.update(resnet_part_params)
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()

    traindataset = PairwiseImg(train='train', inputsize=args.trainsize, data_root=args.data_root, img_path=args.img_path)
    train_loader = data.DataLoader(dataset=traindataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                   num_workers=0, drop_last=True)

    train_total_step = len(train_loader)
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[35, 55], last_epoch=args.start_epoch - 1)

    phases = ['train']
    for epoch in range(args.start_epoch, args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        print('current time {}'.format(datetime.now()))
        train(train_loader, model, optimizer, epoch, train_total_step, phases)
        lr_scheduler.step()
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), args.save_path + '10_GT.pth' + '.%d' % epoch)


def train(train_loader, model, optimizer, epoch, train_total_step, phases):

    train_dataloader = iter(train_loader)
    time = datetime.now()

    for phase in phases:
        if phase == 'train':
            # -------------------------------------train 部分------------------------------------
            model.train()
            for i in range(train_total_step):
                optimizer.zero_grad()
                sup_batch = train_dataloader.next()
                video_image, video_gt, search_img, search_gt,static_img, static_gt = sup_batch['video_image'], sup_batch['video_gt'], \
                                                                                                      sup_batch['search_img'], sup_batch['search_gt'],\
                                                                                                      sup_batch['static_img'], sup_batch['static_label']

                video_image = video_image.cuda()
                video_gt = video_gt.cuda()
                search_img = search_img.cuda()
                search_gt = search_gt.cuda()
                static_img = static_img.cuda()
                static_gt = static_gt.cuda()

                criterion = torch.nn.BCEWithLogitsLoss()

                # ----------------- static image loss -----------------
                if i % 3 == 0:
                    _, static_pre = model(static_img, static_img, static_gt)  # train这里没有传shape参数，所以image多大，out就是多大
                    pre_loss = criterion(static_pre, static_gt)

                    total_loss = pre_loss
                    total_loss.backward()

                    if i % 30 == 0:
                        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], static_pre: {:.4f}'.
                            format(datetime.now(), epoch, args.epochs, i, train_total_step, pre_loss.data))

                # ----------------计算视频损失------------------
                else:
                    search_pre, search_pre_main = model(video_image, search_img, video_gt)  # train这里没有传shape参数，所以image多大，out就是多大
                    pre_loss = criterion(search_pre, search_gt)
                    main_weight = make_weight(search_pre, search_gt)
                    main_loss = weight_loss(search_pre_main, search_gt, main_weight)
                    total_loss = pre_loss + main_loss
                    total_loss.backward()

                    if i % 10 == 0:
                        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], video_pre: {:.4f}'.
                            format(datetime.now(), epoch, args.epochs, i, train_total_step, pre_loss.data))

                optimizer.step()
                if i % 100 == 0 or i == train_total_step:
                    time1 = datetime.now()
                    costtime = time1 - time
                    print('100个iteration计算时间为：：', costtime)
                    time = datetime.now()


if __name__ == '__main__':
    main()



