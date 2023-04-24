import torch
import time
import os
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils import data
from torch.autograd import Variable
from test_dataset import PairwiseImg
from net import SSL_Model

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='1')

    parser.add_argument('-model_name', type=str, default='10GT_50PL_best')
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=448)
    parser.add_argument('-output_stride', type=int, default=16)

    parser.add_argument('-load_path', type=str,
                        default='D:\Bitahub_MODEL\\10GT\\10GT+50PL_best.pth')
    parser.add_argument('-save_dir', type=str, default='.\\eval\\10GT_test')
    parser.add_argument('-test_dataset', type=list, default=['DAVIS', 'FBMS', 'MCL', 'ViSal','SegTrack-V2'])
    #parser.add_argument('-test_dataset', type=list, default=['DAVIS'])
    parser.add_argument('-test_fold', type=str, default='\\test')
    parser.add_argument('--datapath', type=str, default='F:\\pycharm projects2\\MGA\\dataset', help='shujuji')
    parser.add_argument('--img_path', type=str, default='F:\pycharm projects\CPD_initial\TwoDSOD\DUTS-TRAIN',
                        help='shujuji')

    return parser.parse_args()

def ownMinMax(x):
    B, C, H, W = x.shape
    norm_list = []
    for i in range(B):
        img = x[i, 0, :, :]
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


def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)


def sharpen(out):
    inverse = 1 - out
    mul = out * out + inverse * inverse
    return out * out / mul


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = SSL_Model(num_classes=args.num_classes, output_stride=args.output_stride,
                      img_backbone_type='resnet50', backbone_type='resnet50')

    pretrain_weights = torch.load(args.load_path)
    pretrain_keys = list(pretrain_weights.keys())
    pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
    net_keys = list(model.state_dict().keys())
    # ----------新增的----------

    for key in net_keys:
        key_ = 'module.' + key
        if key_ in pretrain_keys:
            assert (model.state_dict()[key].size() == pretrain_weights[key_].size())
            model.state_dict()[key].copy_(pretrain_weights[key_])
        else:
            print('missing key: ', key_)

    print('loaded pre-trained weights.')

    model.cuda()

    for dataset in args.test_dataset:
        print("now dataset {} is tested \n: ".format(dataset))
        testdataset = PairwiseImg(train=False, inputsize=args.input_size, dataset=dataset)
        test_loader = data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

        save_dir = args.save_dir + '/' + args.model_name + '_' + dataset + '/'

        num_iter_ts = len(test_loader)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.eval()

        with torch.no_grad():

            for i, pack in enumerate(test_loader):
                print("progress {}/{}\n".format(i, num_iter_ts))

                inputs, label, flow, shape, image_name = pack


                inputs = Variable(inputs, requires_grad=False)
                inputs = inputs.cuda()
                flow = flow.cuda()
                start = time.time()
                app_out, opt_out, fuse_out = model(inputs, flow)
                end = time.time()



                # -------------------save prediction-----------------------

                OUT = fuse_out
                prob_pred = torch.nn.Sigmoid()(OUT)

                prob_pred = (prob_pred - torch.min(prob_pred) + 1e-8) / (
                            torch.max(prob_pred) - torch.min(prob_pred) + 1e-8)

                prob_pred = F.upsample(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
                save_data = prob_pred[0]
                save_png = save_data[0].numpy()
                save_png = np.round(save_png * 255)
                save_png = save_png.astype(np.uint8)
                save_png = Image.fromarray(save_png)

                save_path = save_dir + image_name[0]
                # a = save_path[:save_path.rfind('\\')]
                # 静态图的话，就是左斜杠，视频图的话，就是右斜杠
                if not os.path.exists(save_path[:save_path.rfind('\\')]):
                    os.makedirs(save_path[:save_path.rfind('\\')])

                save_png.save(save_path)



if __name__ == '__main__':
    args = get_arguments()
    main(args)