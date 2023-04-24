
import torch
import numpy
import os
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

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
    def __init__(self, train='train',
                 inputsize=448,
                 data_root='F:\\pycharm projects2\MGA\\dataset',
                 img_path='F:\pycharm projects\CPD_initial\TwoDSOD\DUTS-TRAIN',
                 seq_name=None,
                 test_data=None):

        self.train = train
        self.inputsize = inputsize
        self.data_root = data_root
        self.img_path = img_path
        self.seq_name = seq_name

        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        if self.train == 'train':
            davis_fname = 'DAVIS\\ImageSets\\2016\\train'
            fbms_fname = 'FBMS\\ImageSets\\train'

        else:
            davis_fname = 'DAVIS\\ImageSets\\2016\\train'
            fbms_fname = 'FBMS\\ImageSets\\train'



        #  --------------------------davis ---------------------------------
        if self.train == 'train':
            with open(os.path.join(data_root, davis_fname + '.txt')) as f:
                seqs = f.readlines()
                video_list = []
                labels = []
                Index = {}

                Annotations = '0-10'  # 10% ground truth

                for seq in seqs:
                    start_num = len(labels)
                    lab = numpy.sort(os.listdir(os.path.join(data_root, 'DAVIS', Annotations, seq.strip('\n'))))
                    lab_path = list(
                        map(lambda x: os.path.join(data_root, 'DAVIS', Annotations, seq.strip(), x), lab))
                    labels.extend(lab_path)
                    end_num = len(labels)
                    Index[seq.strip('\n')] = numpy.array([start_num, end_num])
                    for i in range(len(lab_path)):
                        img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                        img_path = img_path.split('.')[0] + '.jpg'
                        video_list.append(img_path)

            # ----------------------- FBMS --------------------------------
            with open(os.path.join(data_root, fbms_fname + '.txt')) as f:
                seqs = f.readlines()
                Annotations = 'Annotations'

                for seq in seqs:
                    start_num = len(labels)

                    lab = numpy.sort(
                        os.listdir(os.path.join(data_root, 'FBMS', Annotations, seq.strip('\n'))))
                    lab_path = list(
                        map(lambda x: os.path.join(data_root, 'FBMS', Annotations, seq.strip(), x), lab))
                    labels.extend(lab_path)

                    end_num = len(labels)
                    Index[seq.strip('\n')] = numpy.array([start_num, end_num])

                    for i in range(len(lab_path)):
                        img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                        img_path = img_path.split('.')[0] + '.jpg'
                        video_list.append(img_path)


        if self.train == 'test':
            # ----------------------- DAVIS ------------------------------
            if test_data == 'DAVIS':
                with open(os.path.join(data_root, davis_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    Index = {}

                    Annotations = 'Annotations'

                    for seq in seqs:
                        start_num = len(labels)
                        lab = numpy.sort(
                            os.listdir(os.path.join(data_root, 'DAVIS', Annotations, seq.strip('\n'))))
                        lab_path = list(
                            map(lambda x: os.path.join(data_root, 'DAVIS', Annotations, seq.strip(), x),
                                lab))
                        labels.extend(lab_path)
                        end_num = len(labels)
                        Index[seq.strip('\n')] = numpy.array([start_num, end_num])
                        for i in range(len(lab_path)):
                            img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                            img_path = img_path.split('.')[0] + '.jpg'
                            video_list.append(img_path)

            # ------------------------ FBMS ------------------------
            if test_data == 'FBMS':
                with open(os.path.join(data_root, fbms_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    Index = {}
                    for seq in seqs:
                        img = numpy.sort(
                            os.listdir(os.path.join(data_root, 'FBMS', 'JPEGImages', seq.strip('\n'))))
                        img_path = list(
                            map(lambda x: os.path.join(data_root, 'FBMS', 'JPEGImages', seq.strip(), x), img))
                        video_list.extend(img_path)

        # ----------------------------- DUTS -------------------------------
        with open(self.img_path + '/' + 'train.txt', 'r') as lines:  # 读取txt文件，里面都是图片的名字
            static_images = []
            static_labels = []

            for line in lines:
                static_img_path = os.path.join(self.img_path + '/image/' + line.strip() + '.jpg')
                static_images.append(static_img_path)
                static_label_path = os.path.join(self.img_path + '/mask/' + line.strip() + '.png')
                static_labels.append(static_label_path)

        self.static_images = static_images
        self.static_labels = static_labels

        self.video_list = video_list
        self.labels = labels
        self.Index = Index

        self.img_transform = transforms.Compose([
            transforms.Resize((self.inputsize, self.inputsize)),  # 把图片resize成256*256大小
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.inputsize, self.inputsize)),
            transforms.ToTensor()])
        self.colorjitter = transforms.Compose(
            [transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5)])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        img = self.video_list[idx]
        if self.train == 'train':
            label = self.labels[idx]
        label_name = self.video_list[idx].split('\\')[-2] + '\\' + self.video_list[idx].split('\\')[-1]
        label_name = label_name.replace('.jpg', '.png')
        seq_name = img.split("\\")[-2]
        image = cv2.imread(img)[:, :, ::-1].astype(numpy.uint8)
        if self.train == 'train':
            label = cv2.imread(label, 0).astype(numpy.uint8)
        shape = image.shape
        shape = shape[:2]

        img_idx = numpy.random.randint(1, len(self.static_images) - 1)
        static_img = self.static_images[img_idx]
        static_label = self.static_labels[img_idx]
        static_img = cv2.imread(static_img)[:, :, ::-1].astype(numpy.uint8)
        static_label = cv2.imread(static_label, 0).astype(numpy.uint8)

        # --------------训练阶段-------------------
        if self.train == 'train':
            my_index = self.Index[seq_name]
            search_index = [idx - 2, idx + 2]

            if search_index[0] < my_index[0]:
                search_index[0] = idx - 1
                if search_index[0] < my_index[0]:
                    search_index[0] = idx

            if search_index[1] > my_index[1]:
                search_index[1] = idx + 1
                if search_index[1] > my_index[1]:
                    search_index[1] = idx

            search_id = numpy.random.randint(search_index[0], search_index[1])

            if search_id == idx:
                search_id = numpy.random.randint(search_index[0], search_index[1])

            search_img = self.video_list[search_id]
            search_label = self.labels[search_id]
            search_img = cv2.imread(search_img)[:, :, ::-1].astype(numpy.uint8)
            search_label = cv2.imread(search_label, 0).astype(numpy.uint8)

            image, label = self.randomcrop(image, label, None)
            image, label = self.randomflip(image, label, None)

            static_img, static_label = self.randomcrop(static_img, static_label, None)
            static_img, static_label = self.randomflip(static_img, static_label, None)

            search_img, search_label = self.randomcrop(search_img, search_label, None)
            search_img, search_label = self.randomflip(search_img, search_label, None)

            image = Image.fromarray(image)
            label = Image.fromarray(label)

            static_img = Image.fromarray(static_img)
            static_label = Image.fromarray(static_label)

            search_img = Image.fromarray(search_img)
            search_label = Image.fromarray(search_label)

            image = self.img_transform(image)
            label = self.gt_transform(label)
            search_img = self.img_transform(search_img)
            search_label = self.gt_transform(search_label)
            static_img = self.img_transform(static_img)
            static_label = self.gt_transform(static_label)


            sample = {'video_image': image, 'video_gt': label, "search_img": search_img, "search_gt": search_label,
                      "static_img": static_img, "static_label": static_label}
            return sample


        # -------------测试阶段--------------
        if self.train == 'test':

            video_list = []
            video_num = img.split('\\')[-1]

            train_file = img.replace('JPEGImages', '0-10').strip(video_num).strip('\\')  # 这是针对DAVIS数据集的
            train_list = os.listdir(train_file)
            train_list.append(video_num)
            train_list.sort()
            length = len(train_list)
            search_idx = train_list.index(video_num)

            if search_idx == 0:
                video_list.append(train_list[1])
                video_list.append(train_list[2])
            elif search_idx == (length - 1):
                video_list.append(train_list[length - 2])
                video_list.append(train_list[length - 3])
            else:
                video_list.append(train_list[search_idx - 1])
                video_list.append(train_list[search_idx + 1])

            video_gt1 = os.path.join(train_file, video_list[0])
            # video_img1 = video_gt1.replace('0-10', 'JPEGImages')
            video_img1 = video_gt1.replace('0-10', 'JPEGImages').replace('.png', '.jpg')
            video_gt2 = os.path.join(train_file, video_list[1])
            # video_img2 = video_gt2.replace('0-10', 'JPEGImages')
            video_img2 = video_gt2.replace('0-10', 'JPEGImages').replace('.png', '.jpg')

            video_img1 = cv2.imread(video_img1)[:, :, ::-1].astype(numpy.uint8)
            video_gt1 = cv2.imread(video_gt1, 0).astype(numpy.uint8)

            video_img2 = cv2.imread(video_img2)[:, :, ::-1].astype(numpy.uint8)
            video_gt2 = cv2.imread(video_gt2, 0).astype(numpy.uint8)


            image = Image.fromarray(image)
            video_img1 = Image.fromarray(video_img1)
            video_gt1 = Image.fromarray(video_gt1)
            video_img2 = Image.fromarray(video_img2)
            video_gt2 = Image.fromarray(video_gt2)

            image = self.img_transform(image)
            video_img1 = self.img_transform(video_img1)
            video_gt1 = self.gt_transform(video_gt1)
            video_img2 = self.img_transform(video_img2)
            video_gt2 = self.gt_transform(video_gt2)

            sample = {'video_image': image, "search_img1": video_img1, "search_gt1": video_gt1,
                      "search_img2": video_img2, "search_gt2": video_gt2, "shape": shape, "label_name": label_name}
            return sample

