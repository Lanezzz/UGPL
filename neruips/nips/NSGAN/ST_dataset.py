import torch
import h5py
import numpy
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


def func(name, obj):     # function to recursively store all the keys
    global datas, groups
    if isinstance(obj, h5py.Dataset):
        datas.append(name)
    elif isinstance(obj, h5py.Group):
        groups.append(name)


class RandomCrop(object):
    def __call__(self, image, mask, flow):
        H, W, _ = image.shape
        randw = numpy.random.randint(W / 8)
        randh = numpy.random.randint(H / 8)
        offseth = 0 if randh == 0 else numpy.random.randint(randh)
        offsetw = 0 if randw == 0 else numpy.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None and flow is not None:
            return image[p0:p1, p2:p3, :], flow[p0:p1, p2:p3, :]
        elif flow is None and mask is not None:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]
        elif flow is None and mask is None:
            return image[p0:p1, p2:p3, :]
        else:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], flow[p0:p1, p2:p3, :]


class RandomFlip(object):
    def __call__(self, image, mask, flow):
        if numpy.random.randint(2) == 0:
            if mask is None and flow is not None:
                return image[:, ::-1, :].copy(), flow[:, ::-1, :].copy()
            elif flow is None and mask is not None:
                return image[:, ::-1, :].copy(), mask[:, ::-1].copy()
            elif flow is None and mask is None:
                return image[:, ::-1, :].copy()
            else:
                return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), flow[:, ::-1, :].copy()
        else:
            if mask is None and flow is not None:
                return image, flow
            elif flow is None and mask is not None:
                return image, mask
            elif flow is None and mask is None:
                return image
            else:
                return image, mask, flow


class PairwiseImg(Dataset):
    def __init__(self, train=True,
                 inputsize=448,
                 data_root='F:/pycharm projects2/MGA/dataset',
                 img_path='F:/pycharm projects/CPD_initial/TwoDSOD/DUTS-TRAIN',
                 seq_name=None,
                 is_pseudo=False):

        self.train = train
        self.inputsize = inputsize
        self.data_root = data_root
        self.img_path = img_path
        self.seq_name = seq_name
        self.is_pseudo = is_pseudo

        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        # ----------20% labels---------------

        # ----------50% labels---------------
        self.davis_train_num = 1042
        self.fbms_train_num = 3094

        # ----------90% labels---------------

        if self.train:
            davis_fname = 'DAVIS\\ImageSets\\2016\\train'
            fbms_fname = 'FBMS\\ImageSets\\train'
        else:
            davis_fname = 'DAVIS\\ImageSets\\2016\\val'
            fbms_fname = 'FBMS\\ImageSets\\test'


        # ------------------DAVIS--------------------
        with open(os.path.join(data_root, davis_fname + '.txt')) as f:
            seqs = f.readlines()
            video_list = []
            labels = []
            flows = []
            if self.train:
                if self.is_pseudo == False:
                    Annotations = '0-10'
                else:
                    Annotations = '50PL-DF'
            else:
                Annotations = 'Annotations'
            for seq in seqs:
                lab = numpy.sort(os.listdir(os.path.join(data_root, 'DAVIS', Annotations, seq.strip('\n'))))
                lab_path = list(
                    map(lambda x: os.path.join(data_root, 'DAVIS', Annotations, seq.strip(), x), lab))
                labels.extend(lab_path)

                for i in range(len(lab_path)):
                    flow_path = lab_path[i].replace(Annotations, 'raft_flow')
                    flow_path = flow_path.split('.')[0] + '.jpg'
                    flows.append(flow_path)

                    img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                    img_path = img_path.split('.')[0] + '.jpg'
                    video_list.append(img_path)

        #----------------------FBMS-----------------------
        with open(os.path.join(data_root, fbms_fname + '.txt')) as f:
            seqs = f.readlines()

            if self.train:
                if self.is_pseudo == False:
                    Annotations = '0-10'
                else:
                    Annotations = '50PL-DF'
            else:
                Annotations = 'Annotations'

            for seq in seqs:
                lab = numpy.sort(
                    os.listdir(os.path.join(data_root, 'FBMS', Annotations, seq.strip('\n'))))
                lab_path = list(
                    map(lambda x: os.path.join(data_root, 'FBMS', Annotations, seq.strip(), x), lab))
                labels.extend(lab_path)

                for i in range(len(lab_path)):
                    flow_path = lab_path[i].replace(Annotations, 'raft_flow')
                    flow_path = flow_path.split('.')[0] + '.jpg'
                    flows.append(flow_path)

                    img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                    img_path = img_path.split('.')[0] + '.jpg'
                    video_list.append(img_path)

        # ----------------------DUTS-----------------------
        with open(self.img_path + '/' + 'train.txt', 'r') as lines:
            static_images = []
            static_labels = []
            for line in lines:
                static_img_path = os.path.join(self.img_path + '/image/' + line.strip() + '.jpg')
                static_images.append(static_img_path)
                static_label_path = os.path.join(self.img_path + '/mask/' + line.strip() + '.png')
                static_labels.append(static_label_path)

        self.static_images = static_images
        self.static_labels = static_labels

        if self.is_pseudo == False and self.train == True:
            self.unsup_num = self.davis_train_num + self.fbms_train_num
            self.sup_num = len(video_list)
            num1 = self.unsup_num // self.sup_num
            num2 = self.unsup_num % self.sup_num

            self.video_list = video_list * num1
            self.labels = labels * num1
            self.flows = flows * num1

            rand_indices = torch.randperm(len(video_list) - 1).tolist()
            new_indices = rand_indices[:num2]

            self.video_list += [video_list[i] for i in new_indices]
            self.labels += [labels[i] for i in new_indices]
            self.flows += [flows[i] for i in new_indices]
        else:

            self.video_list = video_list
            self.labels = labels
            self.flows = flows


        self.img_transform = transforms.Compose([
            transforms.Resize((self.inputsize, self.inputsize)),  # 把图片resize成256*256大小
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.inputsize, self.inputsize)),
            transforms.ToTensor()])
        self.colorjitter = transforms.Compose(
            [transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.4)])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        img = self.video_list[idx]
        label = self.labels[idx]
        video_flow = self.flows[idx]

        image = cv2.imread(img)[:, :, ::-1].astype(numpy.uint8)
        label = cv2.imread(label, 0).astype(numpy.uint8)
        flow = cv2.imread(video_flow)[:, :, ::-1].astype(numpy.uint8)

        image, label, flow = self.randomcrop(image, label, flow)
        image, label, flow = self.randomflip(image, label, flow)

        image = Image.fromarray(image)
        label = Image.fromarray(label)
        flow = Image.fromarray(flow)

        image = self.img_transform(image)
        label = self.gt_transform(label)
        flow = self.img_transform(flow)

        if self.is_pseudo == False:
            img_idx1 = numpy.random.randint(1, len(self.static_images) - 1)

            static_img1 = self.static_images[img_idx1]
            static_label1 = self.static_labels[img_idx1]
            # static_name = static_img1.split('/')[-1].split('.')[0] + '.png'

            static_img1 = cv2.imread(static_img1)[:, :, ::-1].astype(numpy.uint8)
            static_label1 = cv2.imread(static_label1, 0).astype(numpy.uint8)
            static_flow1 = numpy.zeros(static_img1.shape, dtype=numpy.uint8)

            img_idx2 = numpy.random.randint(1, len(self.static_images) - 1)

            static_img2 = self.static_images[img_idx2]
            static_label2 = self.static_labels[img_idx2]

            static_img2 = cv2.imread(static_img2)[:, :, ::-1].astype(numpy.uint8)
            static_label2 = cv2.imread(static_label2, 0).astype(numpy.uint8)
            static_flow2 = numpy.zeros(static_img2.shape, dtype=numpy.uint8)

            static_img1, static_label1, static_flow1 = self.randomcrop(static_img1, static_label1, static_flow1)
            static_img1, static_label1, static_flow1 = self.randomflip(static_img1, static_label1, static_flow1)

            static_img1 = Image.fromarray(static_img1)
            static_label1 = Image.fromarray(static_label1)
            static_flow1 = Image.fromarray(static_flow1)

            static_img1 = self.img_transform(static_img1)
            static_label1 = self.gt_transform(static_label1)
            static_flow1 = self.img_transform(static_flow1)

            static_img2, static_label2, static_flow2 = self.randomcrop(static_img2, static_label2, static_flow2)
            static_img2, static_label2, static_flow2 = self.randomflip(static_img2, static_label2, static_flow2)

            static_img2 = Image.fromarray(static_img2)
            static_label2 = Image.fromarray(static_label2)
            static_flow2 = Image.fromarray(static_flow2)

            static_img2 = self.img_transform(static_img2)
            static_label2 = self.gt_transform(static_label2)
            static_flow2 = self.img_transform(static_flow2)

            sample = {'video_image': image, 'video_gt': label, 'video_flow': flow,
                      'static_img1': static_img1, 'static_gt1': static_label1, 'static_flow1': static_flow1,
                      'static_img2': static_img2, 'static_gt2': static_label2, 'static_flow2': static_flow2}

        else:
            sample = {'video_image': image, 'video_gt': label, 'video_flow': flow}

        # sample = {'video_image': image, 'video_gt': label, 'video_flow': flow}
        return sample





