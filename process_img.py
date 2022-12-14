import os
from PIL import Image
import torch
import torchvision
import utils
import random
label_root = "/home/lwx/HOG/data/csgo225labeled/labels"
img_root = "/home/lwx/HOG/data/csgo225labeled/images"
crop_path = "/home/lwx/HOG/data/csgo225labeled/postives"
neg_path = "/home/lwx/HOG/data/csgo225labeled/negtives"
def get_raw_pos_and_neg():
    if os.path.exists(crop_path):
        os.system("rm -r " + crop_path)
    os.system("mkdir " + crop_path)
    if os.path.exists(neg_path):
        os.system("rm -r " + neg_path)
    os.system("mkdir " + neg_path)
    imgs = os.listdir(img_root)
    labels = os.listdir(label_root)
    imgs.sort()
    labels.sort()
    # print(len(imgs), len(labels))
    # for idx in range(0, len(imgs)):
    #     item = imgs[idx].split('.')[0]
    #     item = item + ".txt"
    #     if(item not in labels):
    #         print(item, imgs[idx])
    number = 0
    final_label_text = []
    hs = []
    ws = []
    for idx in range(0, len(labels)):
        label_path = os.path.join(label_root, labels[idx])
        with open(label_path, 'r') as f:
            coor = f.readlines()
        img_path = os.path.join(img_root, imgs[idx])
        img = Image.open(img_path)
        W, H = img.size

        for x in coor:
            if (x[0] == '2' or x[0] == '3'):
                label_float = x.rstrip().split(' ')[1:]
                x0 = float(label_float[0]) * W
                y0 = float(label_float[1]) * H
                w = float(label_float[2]) * W
                h = float(label_float[3]) * H
                bbox = [x0, y0, w, h]
                bbox = utils._box_cxcywh_to_xyxy(torch.as_tensor(bbox))

                top = int(bbox[1])
                left = int(bbox[0])
                h = int(bbox[3] - bbox[1])
                w = int(bbox[2] - bbox[0])

                hs.append(h)
                ws.append(w)

                pos_img = torchvision.transforms.functional.crop(img, top, left, h, w)
                pos_img.save(crop_path + "/" + "p" +str(number) + ".jpg")
                final_label_text.append(crop_path + "/" + "p" +str(number) + ".jpg " + "1\n")

                n_top = random.randint(0, H - h - 1)
                n_left = random.randint(0, W - w - 1)
                neg_img = torchvision.transforms.functional.crop(img, n_top, n_left, h, w)
                neg_img.save(neg_path + "/" + "n" + str(number) + ".jpg")

                final_label_text.append(neg_path + "/" + "n" + str(number) + ".jpg " + "0\n")
                number += 1
    with open('/home/lwx/HOG/data/csgo225labeled/new_label.txt','w') as f:
        f.write(''.join(final_label_text))
    print(max(ws), min(ws), sum(ws) / len(ws))
    print(max(hs), min(hs), sum(hs) / len(hs))

def img_reshape():
    with open('/home/lwx/HOG/data/csgo225labeled/new_label.txt', 'r') as f:
        items = f.readlines()
    items = [x.split(' ')[0] for x in items]
    for idx in range(0, len(items)):
        img = Image.open(items[idx])
        img = img.resize((56, 56 * 2), Image.Resampling.LANCZOS)
        img.save(items[idx])
img_reshape()