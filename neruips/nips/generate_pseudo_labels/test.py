import argparse
import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils import data
from dataset import PairwiseImg
from model import CoattentionModel


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0')

    parser.add_argument('-model_name', type=str, default='UGPLG')
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=448)
    parser.add_argument('-output_stride', type=int, default=16)

    parser.add_argument('-load_path', type=str,
                        default='F:\pycharm projects2\OwnCos\save_output\\2path_weight_bce\\10_GT_2_paths.pth.69')

    parser.add_argument('-save_dir', type=str, default='./eval')

    parser.add_argument('-test_dataset', type=str, default='FBMS', choices=['DAVIS', 'FBMS'])
    parser.add_argument('-test_fold', type=str, default='/train')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    net = CoattentionModel(num_classes=1, all_channel=256, all_dim=56 * 56)

    # load pre-trained weights
    pretrain_weights = torch.load(args.load_path)
    pretrain_keys = list(pretrain_weights.keys())
    pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
    net_keys = list(net.state_dict().keys())

    for key in net_keys:
        key_ = 'module.' + key
        if key_ in pretrain_keys:
            assert (net.state_dict()[key].size() == pretrain_weights[key_].size())
            net.state_dict()[key].copy_(pretrain_weights[key_])
        else:
            print('missing key: ', key_)
    print('loaded pre-trained weights.')

    net.cuda()

    testdataset = PairwiseImg(train='test', inputsize=args.input_size, test_data=args.test_dataset)
    test_loader = data.DataLoader(dataset=testdataset, batch_size=1,shuffle=False)

    save_dir = args.save_dir + args.test_fold + '-' + args.model_name + '-' + args.test_dataset + '/saliency_map/'
    num_iter_ts = len(test_loader)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.eval()

    with torch.no_grad():
        for i, pack in enumerate(test_loader):
            print("progress {}/{}\n".format(i, num_iter_ts))

            search_image1, search_gt1, search_image2, search_gt2, video_img,shape,image_name = pack['search_img1'], \
                                                                                      pack['search_gt1'], \
                                                                                      pack['search_img2'], \
                                                                                      pack['search_gt2'], \
                                                                                      pack['video_image'], \
                                                                                      pack['shape'], \
                                                                                      pack['label_name']

            search_image1 = search_image1.cuda()
            search_gt1 = search_gt1.cuda()
            video_img = video_img.cuda()

            val_out1, val_out_main1 = net(search_image1, video_img, search_gt1)  # train这里没有传shape参数，所以image多大，out就是多大

            # --------------------------存图---------------------------
            val_out = torch.nn.Sigmoid()(val_out_main1)
            val_out = (val_out - torch.min(val_out) + 1e-8) / (torch.max(val_out) - torch.min(val_out) + 1e-8)

            val_out = F.upsample(val_out, size=shape, mode='bilinear', align_corners=True).cpu().data
            save_data = val_out[0]
            save_png = save_data[0].numpy()
            save_png = np.round(save_png * 255)
            save_png = save_png.astype(np.uint8)
            save_png = Image.fromarray(save_png)

            save_path = save_dir + image_name[0]
            if not os.path.exists(save_path[:save_path.rfind('\\')]):
                os.makedirs(save_path[:save_path.rfind('\\')])
            save_png.save(save_path)

if __name__ == '__main__':
    args = get_arguments()
    main(args)