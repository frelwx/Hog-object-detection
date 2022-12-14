import torch
import torchvision
from PIL import Image
import cv2
from skimage.feature import hog
import random
def random_flip(img, p=0.5):
    if(random.random() < p):
        img = cv2.flip(img, 1)
    return img
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
    transforms = random_flip
    dataset = recognition_dataset(transforms=transforms)
    fd, label = dataset[208]
    print(type(fd), label, fd.shape)
    # img = torchvision.transforms.ToPILImage()(img)
    # img.save("tmpp.jpg")

    

