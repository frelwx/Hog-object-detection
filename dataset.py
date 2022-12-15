import torch
import torchvision
from PIL import Image
import cv2
from skimage.feature import hog
import random
import os
import utils
import iou
def random_flip(img, p=0.5):
    if(random.random() < p):
        img = cv2.flip(img, 1)
    return img

class detection_dataset(torch.utils.data.Dataset):
    def __init__(self, root="/home/lwx/HOG/data/csgo225labeled", transforms=None):
        super().__init__()
        self.imgs = os.listdir(os.path.join(root, "images"))
        self.lables = os.listdir(os.path.join(root, "labels"))
        self.root = root
        self.imgs.sort()
        self.lables.sort()
    def __len__(self):
        # return len(self.imgs)
        return 1
    def get_feature(self, img):
        fd = hog(img, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), channel_axis=-1)
        fd = torch.from_numpy(fd)
        fd = fd.to(dtype=torch.float32)
        return fd
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = cv2.imread(img_path)
        coor_path = os.path.join(self.root, "labels", self.lables[idx])
        with open(coor_path, 'r') as f:
            coor = f.readlines()
        H, W = img.shape[0], img.shape[1]
        x_bias = 5
        y_bias = 5
        new_w = 56
        new_h = new_w * 2
        samples = []
        lables = []
        xy = []
        for x in coor:
            if (x[0] == '2' or x[0] == '3'):
                label_float = x.rstrip().split(' ')[1:]
                x0 = float(label_float[0]) * W
                y0 = float(label_float[1]) * H
                w = float(label_float[2]) * W
                h = float(label_float[3]) * H
                bbox = [x0, y0, w, h]
                cnt = 0
                for xb in range(-x_bias, x_bias + 1):
                    for yb in range(-y_bias, y_bias + 1):
                        new_x_0 = min(max(0, x0 + xb), W - 1)
                        new_y_0 = min(max(0, y0 + yb), H - 1)
                        bbox = [new_x_0, new_y_0, new_w, new_h]
                        bbox = utils._box_cxcywh_to_xyxy(torch.as_tensor(bbox)).to(dtype=torch.int32)
                        if bbox[0] < 0 or bbox[0] >= W or bbox[2] < 0 or bbox[2] >= W:
                            continue
                        if bbox[1] < 0 or bbox[1] >= H or bbox[3] < 0 or bbox[3] >= H:
                            continue
                        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                        samples.append(self.get_feature(crop_img))
                        lables.append(0)
                        xy.append((new_x_0, new_y_0))
                        # cv2.imwrite("./test/" + str(bbox[0]) + "_" + str(bbox[1]) + ".jpg", crop_img)
                        flag = True
                        while(flag):
                            flag = False
                            n_top = random.randint(0, H - new_h - 1)
                            n_left = random.randint(0, W - new_w - 1)
                            y1 = n_top + new_h
                            x1 = n_left + new_w
                            tmp_bbox = torch.as_tensor([n_left, n_top, x1, y1])
                            neg_iou = iou.iou(bbox.unsqueeze(dim=0), tmp_bbox.unsqueeze(dim=0))
                            if(neg_iou.item() > 0.5):
                                # print("here")
                                flag = True
                        neg_img = img[n_top : n_top + new_h, n_left : n_left + new_w, :]
                        samples.append(self.get_feature(neg_img))
                        lables.append(1)
                        tmp_bbox = utils._box_xyxy_to_cxcywh(tmp_bbox).numpy()
                        xy.append((tmp_bbox[0], tmp_bbox[1]))
                        cnt += 1
        if(len(samples) < 10):
            print(idx, len(samples), self.lables[idx])
        return torch.stack(samples, dim=0), torch.as_tensor(lables, dtype=torch.long), xy

        
class recognition_dataset(torch.utils.data.Dataset):
    def __init__(self, split_path="/home/lwx/HOG/data/csgo225labeled/new_label.txt", transforms=None):
        super().__init__()
        with open(split_path, 'r') as f:
            self.split_list = f.readlines()
        self.split_list = [x.strip('\n') for x in self.split_list]
        self.split_list.sort()
        self.transforms = transforms
    def __len__(self):
        return len(self.split_list)
    def __getitem__(self, idx):
        item = self.split_list[idx]
        path, label = item.split(' ')
        label = torch.tensor(int(label), dtype=torch.int64)
        # img = Image.open(path)
        img = cv2.imread(path)
        if self.transforms != None:
            img = self.transforms(img)
        fd = hog(img, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), channel_axis=-1)
        fd = torch.from_numpy(fd)
        fd = fd.to(dtype=torch.float32)
        return fd, label

if __name__ == "__main__":
    # transforms = random_flip
    # dataset = recognition_dataset(transforms=transforms)
    # fd, label = dataset[208]
    # print(type(fd), label, fd.shape)
    # img = torchvision.transforms.ToPILImage()(img)
    # img.save("tmpp.jpg")
    dataset = detection_dataset()
    imgs, labels, xy = dataset[0]
    print(imgs.shape, len(labels), xy)

    

