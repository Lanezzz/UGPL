
import numpy
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

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
                 dataset = 'DAVIS',
                 data_root='F:\\pycharm projects2\MGA\\dataset',
                 img_path='F:\pycharm projects\CPD_initial\TwoDSOD\DUTS-TRAIN',
                 seq_name=None,
                 davis_ratio=10,
                 davsod_ratio=10,
                 fbms_ratio=1):

        self.train = train
        self.inputsize = inputsize
        self.data_root = data_root
        self.img_path = img_path
        self.seq_name = seq_name
        self.dataset = dataset

        self.davis_ratio = davis_ratio
        self.davsod_ratio = davsod_ratio
        self.fbms_ratio = fbms_ratio

        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        self.davsod_train_num = 7183
        self.davis_train_num = 2079
        self.fbms_train_num = 6554
        # self.VOS_train_num = 66897

        if self.train:
            davis_fname = 'DAVIS\\ImageSets\\2016\\train'
            davsod_fname = 'DAVSOD\\ImageSets\\train'
            fbms_fname = 'FBMS\\ImageSets\\train'
        else:
            davis_fname = 'DAVIS\\ImageSets\\2016\\val'
            davsod_fname = 'DAVSOD\\ImageSets\\test'
            fbms_fname = 'FBMS\\ImageSets\\test'
            visal_fname = 'ViSal\\ImageSets\\test'
            SegV2_fname = 'SegTrack-V2\\ImageSets\\test'
            VOS_fname = 'VOS\\ImageSets\\test'
            MCL_fname = 'MCL\\ImageSets\\test'

        if self.seq_name is None:  # 所有的数据集都参与训练
            #  --------------------------davis 数据集分配---------------------------------
            if self.dataset == 'DAVIS':
                with open(os.path.join(data_root, davis_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    flows = []
                    if self.train:
                        Annotations = '0-' + str(self.davis_ratio)
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

            # ----------------------FBMS数据集分配--------------------------------
            if self.dataset == 'FBMS':
                with open(os.path.join(data_root, fbms_fname + '.txt')) as f:
                    seqs = f.readlines()
                    if self.train==False:
                        video_list = []
                        labels = []
                        flows = []
                    if self.train:
                        Annotations = 'Annotations'
                    else:
                        Annotations = 'Annotations'

                    for seq in seqs:
                        if self.train:
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

                        else:
                            Annotations = 'Annotations'
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



            # ---------------------visal数据集分配------------------------------------
            if self.dataset == 'ViSal':
                with open(os.path.join(data_root, visal_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    flows = []
                    if self.train:
                        Annotations = '0-' + str(self.davis_ratio)
                    else:
                        Annotations = 'Annotations'
                    for seq in seqs:
                        lab = numpy.sort(os.listdir(os.path.join(data_root, 'ViSal', Annotations, seq.strip('\n'))))
                        lab_path = list(
                            map(lambda x: os.path.join(data_root, 'ViSal', Annotations, seq.strip(), x), lab))
                        labels.extend(lab_path)

                        for i in range(len(lab_path)):
                            flow_path = lab_path[i].replace(Annotations, 'raft_flow')
                            flow_path = flow_path.split('.')[0] + '.jpg'
                            flows.append(flow_path)

                            img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                            img_path = img_path.split('.')[0] + '.jpg'
                            video_list.append(img_path)



            # ---------------------MCL数据集分配------------------------------------
            if self.dataset == 'MCL':
                with open(os.path.join(data_root, MCL_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    flows = []
                    if self.train:
                        Annotations = '0-' + str(self.davis_ratio)
                    else:
                        Annotations = 'Annotations'
                    for seq in seqs:
                        lab = numpy.sort(os.listdir(os.path.join(data_root, 'MCL', Annotations, seq.strip('\n'))))
                        lab_path = list(
                            map(lambda x: os.path.join(data_root, 'MCL', Annotations, seq.strip(), x), lab))
                        labels.extend(lab_path)

                        for i in range(len(lab_path)):
                            flow_path = lab_path[i].replace(Annotations, 'raft_flow')
                            flow_path = flow_path.split('.')[0] + '.jpg'
                            flows.append(flow_path)

                            img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                            img_path = img_path.split('.')[0] + '.png'
                            video_list.append(img_path)

            #--------------------------SegTrack-V2------------------------------

            if self.dataset == 'SegTrack-V2':
                with open(os.path.join(data_root, SegV2_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    flows = []
                    if self.train:
                        Annotations = '0-' + str(self.davis_ratio)
                    else:
                        Annotations = 'Annotations'
                    for seq in seqs:
                        lab = numpy.sort(os.listdir(os.path.join(data_root, 'SegTrack-V2', Annotations, seq.strip('\n'))))
                        lab_path = list(
                            map(lambda x: os.path.join(data_root, 'SegTrack-V2', Annotations, seq.strip(), x), lab))
                        labels.extend(lab_path)

                        for i in range(len(lab_path)):
                            flow_path = lab_path[i].replace(Annotations, 'raft_flow')
                            flow_path = flow_path.split('.')[0] + '.jpg'
                            flows.append(flow_path)

                            img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                            img_path = img_path.split('.')[0] + '.jpg'
                            video_list.append(img_path)



            #---------------------------DAVSOD------------------------------

            if self.dataset == 'DAVSOD':
                with open(os.path.join(data_root, davsod_fname + '.txt')) as f:
                    seqs = f.readlines()
                    video_list = []
                    labels = []
                    flows = []
                    #confidence = []
                    if self.train:
                        Annotations = '0-' + str(self.davsod_ratio)
                        #Annotations = '0-' + str(self.davis_ratio) + '+20PL-BCE'
                    else:
                        Annotations = 'Annotations'
                    for seq in seqs:
                        lab = numpy.sort(os.listdir(os.path.join(data_root, 'DAVSOD', Annotations, seq.strip('\n'))))
                        lab_path = list(
                            map(lambda x: os.path.join(data_root, 'DAVSOD', Annotations, seq.strip(), x), lab))
                        labels.extend(lab_path)

                        for i in range(len(lab_path)):
                            flow_path = lab_path[i].replace(Annotations, 'raft_flow')
                            flow_path = flow_path.split('.')[0] + '.jpg'
                            flows.append(flow_path)

                            img_path = lab_path[i].replace(Annotations, 'JPEGImages')
                            img_path = img_path.split('.')[0] + '.png'
                            video_list.append(img_path)



            # -----------------------------DUTS数据集分配-------------------------------

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

            img_idx = numpy.random.randint(1, len(self.static_images) - 1)

            static_img = self.static_images[img_idx]
            static_label = self.static_labels[img_idx]

            image = cv2.imread(img)[:, :, ::-1].astype(numpy.uint8)
            label = cv2.imread(label, 0).astype(numpy.uint8)
            flow = cv2.imread(video_flow)[:, :, ::-1].astype(numpy.uint8)

            static_img = cv2.imread(static_img)[:, :, ::-1].astype(numpy.uint8)
            static_label = cv2.imread(static_label, 0).astype(numpy.uint8)
            static_flow = numpy.zeros(static_img.shape, dtype=numpy.uint8)



            if self.train == True:

                image, label, flow = self.randomcrop(image, label, flow)
                image, label, flow = self.randomflip(image, label, flow)

                static_img, static_label, static_flow = self.randomcrop(static_img, static_label,static_flow)
                static_img, static_label, static_flow = self.randomflip(static_img, static_label,static_flow)

                image = Image.fromarray(image)
                label = Image.fromarray(label)
                flow = Image.fromarray(flow)

                static_img = Image.fromarray(static_img)
                static_label = Image.fromarray(static_label)
                static_flow = Image.fromarray(static_flow)


                image = self.colorjitter(image)
                static_img = self.colorjitter(static_img)

                image = self.img_transform(image)
                label = self.gt_transform(label)
                flow = self.img_transform(flow)


                static_img = self.img_transform(static_img)
                static_label = self.gt_transform(static_label)
                static_flow = self.img_transform(static_flow)


                sample = {'video_image': image, 'video_gt': label, 'video_flow': flow,
                          'static_img': static_img, 'static_gt': static_label, 'static_flow': static_flow}
                return sample

            else:
                label_name = self.labels[idx].split('\\')[-2] + '\\' + self.labels[idx].split('\\')[-1]
                label_name = label_name.split('.')[0] + '.png'
                shape = image.shape
                shape = shape[:2]

                image = Image.fromarray(image)
                label = Image.fromarray(label)
                flow = Image.fromarray(flow)

                image = self.img_transform(image)
                label = self.gt_transform(label)
                flow = self.img_transform(flow)

                return image, label, flow, shape, label_name




