import os, argparse
import torch
import torch.nn as nn
import numpy as np

from datetime import datetime
from torch.utils import data
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from net import SSL_Model
from discriminator import FCDiscriminator
from ST_dataset import PairwiseImg

parser = argparse.ArgumentParser()
parser.add_argument('--lr_G', type=float, default=0.015, help='learning rate')
parser.add_argument('--lr_D', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--start-epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
parser.add_argument('--save_path', type=str, default='/output',
                    help='The directory used to save the trained models')

parser.add_argument('--data_root', type=str, default='F:/pycharm projects2/MGA/dataset', help='shujuji')
parser.add_argument('--img_path', type=str, default='F:/pycharm projects/CPD_initial/TwoDSOD/DUTS-TRAIN', help='shujuji')

parser.add_argument('--mode', type=str, default='train', help='train mode')
parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run')
parser.add_argument('--appearance_restore_from', type=str,
                    default='F:/pycharm projects2/MGA/save_output/pretrain_resnet50.pth',
                    help='the pretrained resnet-50 on DUTS')
parser.add_argument('--motion_restore_from', type=str,
                    default='F:/pycharm projects2/MGA/pre_train/resnet50-19c8e357.pth',
                    help='the pretrained resnet-50 on 20% davis and davsod')

parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--output_stride', type=int, default=16)
parser.add_argument('--davis_ratio', type=int, default=10)
parser.add_argument('--davsod_ratio', type=int, default=10)
parser.add_argument('--fbms_ratio', type=int, default=1)


def unCriterion(pred, label, mask):

    weit = mask
    criterion = torch.nn.BCEWithLogitsLoss(weight=weit).cuda()

    return criterion(pred, label)


def make_D_label(label, ignore_mask):
    D_label = np.ones(ignore_mask.shape) * label
    D_label[ignore_mask] = 1
    D_label = Variable(torch.FloatTensor(D_label)).cuda()
    return D_label


args = parser.parse_args()
def main():
    modelG = SSL_Model(num_classes=args.num_classes, output_stride=args.output_stride,
                       img_backbone_type='resnet50', backbone_type='resnet50')
    modelD = FCDiscriminator()
    modelD = torch.nn.DataParallel(modelD).cuda()

    print('Load pretrained ', args.appearance_restore_from, '...')
    print('Load pretrained ', args.motion_restore_from, '...')

    app_pretrain_dict = torch.load(args.appearance_restore_from)
    mot_pretrain_dict = torch.load(args.motion_restore_from)
    new_app = {}
    new_mot = {}

    for k in app_pretrain_dict.keys():
        new_key = 'app_branch.' + k[7:]
        new_app[new_key] = app_pretrain_dict[k]

    for k in mot_pretrain_dict.keys():
        new_key = 'mot_branch.' + k
        new_mot[new_key] = mot_pretrain_dict[k]

    # --------------end--------------------
    model_params = {}
    state_dict = modelG.state_dict()
    for k, v in state_dict.items():
        if k in new_app.keys():
            v = new_app[k]
            model_params[k] = v
        elif k in new_mot.keys():
            v = new_mot[k]
            model_params[k] = v
        else:
            print('missing keys :', k)

    state_dict.update(model_params)

    modelG.load_state_dict(state_dict)
    modelG = torch.nn.DataParallel(modelG).cuda()


    ## dataset
    traindataset = PairwiseImg(train=True, inputsize=args.trainsize, is_pseudo=False, data_root=args.data_root, img_path=args.img_path)
    train_loader = data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                   num_workers=0, drop_last=True)

    undataset = PairwiseImg(train=True, inputsize=args.trainsize, is_pseudo=True)
    un_loader = data.DataLoader(undataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                num_workers=0, drop_last=True)

    paramsG = modelG.parameters()
    paramsD = modelD.parameters()
    optimizerG = torch.optim.SGD(paramsG, lr=args.lr_G, momentum=args.momentum, weight_decay=args.weight_decay,
                                 nesterov=True)
    optimizerD = torch.optim.Adam(paramsD, args.lr_D)

    lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizerG,
                                                          milestones=[20, 40], last_epoch=args.start_epoch - 1)
    lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizerD,
                                                          milestones=[20, 40], last_epoch=args.start_epoch - 1)

    for epoch in range(args.start_epoch, args.epochs):
        print('current lr_G {:.5e}'.format(optimizerG.param_groups[0]['lr']))
        print('current lr_D {:.5e}'.format(optimizerD.param_groups[0]['lr']))

        print('current time {}'.format(datetime.now()))

        train(train_loader, un_loader, modelG, modelD, optimizerG, optimizerD, epoch)

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if epoch % 49 == 0:
            D_path = os.path.join(args.save_path, '10GT_GAN_modelD.pth' + '.%d' % epoch)
            G_path = os.path.join(args.save_path, '10GT_20PL_GAN.pth' + '.%d' % epoch)
            torch.save(modelG.state_dict(), G_path)
            torch.save(modelD.state_dict(), D_path)


def train(train_loader, un_loader, modelG, modelD, optimizerG, optimizerD, epoch):

    total_step = len(train_loader)
    sup_dataloader = iter(train_loader)
    un_loader = iter(un_loader)
    modelG.train()
    modelD.train()
    time = datetime.now()

    for i in range(total_step):

        optimizerG.zero_grad()
        optimizerD.zero_grad()

        sup_batch = sup_dataloader.next()
        un_batch = un_loader.next()

        #------stop to train the discriminator---------
        for param in modelD.parameters():
            param.requires_grad = False

        video_image, video_gt, video_flow, static_img1, static_gt1, static_flow1, \
        static_img2, static_gt2, static_flow2 = sup_batch['video_image'], \
                                                sup_batch['video_gt'], \
                                                sup_batch['video_flow'], \
                                                sup_batch['static_img1'], \
                                                sup_batch['static_gt1'], \
                                                sup_batch['static_flow1'], \
                                                sup_batch['static_img2'], \
                                                sup_batch['static_gt2'], \
                                                sup_batch['static_flow2']

        video_image = video_image.cuda()
        video_gt = video_gt.cuda()
        video_flow = video_flow.cuda()

        static_img1 = static_img1.cuda()
        static_gt1 = static_gt1.cuda()
        static_flow1 = static_flow1.cuda()

        static_img2 = static_img2.cuda()
        static_gt2 = static_gt2.cuda()
        static_flow2 = static_flow2.cuda()

        un_image, un_gt, un_flow = un_batch['video_image'], un_batch['video_gt'], un_batch['video_flow']

        un_image = un_image.cuda()
        un_gt = un_gt.cuda()
        un_flow = un_flow.cuda()

        criterion = torch.nn.BCEWithLogitsLoss().cuda()

        if i % 3 == 0:
            # ---------static img---------
            static_pred1 = modelG(static_img1, static_flow1, True)
            static_ce_loss1 = criterion(static_pred1, static_gt1)

            static_pred2 = modelG(static_img2, static_flow2, True)
            static_ce_loss2 = criterion(static_pred2, static_gt2)

            static_loss = 0.8 * (static_ce_loss1 + static_ce_loss2)
            static_loss.backward()

            print('       image data    Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], total_static_loss: {:.4f}'
                  .format(epoch, args.epochs, i, total_step, static_loss.data))

            # ---------video data---------
        else:
            video_app, video_opt, video_pred = modelG(video_image, video_flow)
            video_ce_loss = criterion(video_pred, video_gt)
            app_ce_loss = criterion(video_app, video_gt)
            opt_ce_loss = criterion(video_opt, video_gt)

            video_D_out = modelD(video_pred)
            video_gt_ignore_mask = (video_gt.data.cpu().numpy() == 1)  # 显著性部分才会被置为True
            video_lossG_adv = criterion(video_D_out,
                                        make_D_label(1, video_gt_ignore_mask))  # 这边的判别器只是提供梯度给生成器，自己是不更新参数的

            # ---------------pseudo label-----------------------

            un_app, un_opt, un_pred = modelG(un_image, un_flow)
            unsup_gt_D_out = modelD(un_gt, 1)
            unsup_gt_D_out_sigmoid = nn.Sigmoid()(unsup_gt_D_out)

            un_video_ce_loss = unCriterion(un_pred, un_gt, unsup_gt_D_out_sigmoid)
            un_app_ce_loss = unCriterion(un_app, un_gt, unsup_gt_D_out_sigmoid)
            un_opt_ce_loss = unCriterion(un_opt, un_gt, unsup_gt_D_out_sigmoid)

            video_loss = video_ce_loss + app_ce_loss + 0.2 * opt_ce_loss + 0.1 * video_lossG_adv
            un_video_loss = un_video_ce_loss + un_app_ce_loss + 0.2 * un_opt_ce_loss
            final_loss = video_loss + un_video_loss
            final_loss.backward()

            print(
                '      video data     Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], app_loss: {:.4f}, opt_loss: {:.4f},fuse_loss: {:.4f}, video_loss: {:.4f}'
                    .format(epoch, args.epochs, i, total_step, app_ce_loss.data, opt_ce_loss.data, video_ce_loss.data,
                            video_loss.data))

        #------start to train the discriminator---------

        for param in modelD.parameters():
            param.requires_grad = True

        modelD_video_gt = sup_batch['video_gt']
        modelD_video_gt = modelD_video_gt.cuda()

        if i % 3 != 0:
            video_pred_remain = video_pred.detach()  # 把生成器关了
            video_pre_D_out = modelD((video_pred_remain))
            loss_D1 = criterion(video_pre_D_out, make_D_label(0, video_gt_ignore_mask))  # loss_D

            video_ignore_mask_gt = (modelD_video_gt.data.cpu().numpy() == 1)
            video_gt = modelD_video_gt.float()
            video_gt_D_out = modelD(video_gt, 1)
            loss_D2 = criterion(video_gt_D_out, make_D_label(1, video_ignore_mask_gt))  # loss_D

            video_lossdis = (loss_D1 + loss_D2) / 2

            loss_dis = video_lossdis
            # video_lossdis.backward()
            loss_dis.backward()


            print('Discriminitor  Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], video_lossdis: {:.4f}'.
                  format(epoch, args.epochs, i, total_step, video_lossdis.data))

        optimizerG.step()
        optimizerD.step()



        if i % 100 == 0 or i == total_step:
            time1 = datetime.now()
            costtime = time1 - time
            print('100个iteration计算时间为：：', costtime)
            time = datetime.now()

            # time5 = datetime.now()
            # spendtime4 = (time5-time4)

if __name__ == '__main__':
    main()