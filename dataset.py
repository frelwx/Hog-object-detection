import torch
import torchvision
from PIL import Image
class detect_dataset(torch.utils.data.Dataset):
    def __init__(self, root="F:/CCPD2019.tar/CCPD2019", split_path="F:/CCPD2019.tar/CCPD2019/splits/train.txt", transforms=None):
        super().__init__()
        with open(split_path, 'r') as f:
            self.split_list = f.readlines()
        self.split_list = [x.strip('\n') for x in self.split_list]
        self.split_list.sort()
        self.split_list = self.split_list[:]
        self.root = root
        self.transforms = transforms
    def __len__(self):
        return len(self.split_list)
    def __getitem__(self, idx):
        item_path = self.split_list[idx]
        img = Image.open(self.root + '/' + item_path)
        label_inform = item_path.split('/')[-1].rsplit('.', 1)[0].split('-')

        bbox = []
        bbox_inform = [x.split('&') for x in label_inform[2].split('_')]
        bbox = [float(bbox_inform[0][0]), float(bbox_inform[0][1]), float(bbox_inform[1][0]), float(bbox_inform[1][1])]
        bbox  = torch.as_tensor(bbox).unsqueeze(dim=0)

        # numbers = label_inform[-3].split('_')
        # numbers = [int(x) for x in numbers]
        # numbers = torch.as_tensor(numbers, dtype=torch.long)
        if self.transforms != None:
            img, bbox = self.transforms(img, bbox)
        else:
            img = torchvision.transforms.ToTensor()(img)

        return img, bbox
if __name__ == "__main__":
    from transforms import Compose, ToTensor, Normalize, Resize
    transforms = Compose([ToTensor()])
    dataset = detect_dataset(transforms=transforms, split_path="F:/CCPD2019.tar/CCPD2019/splits/train.txt")
    img, boxes, _ = dataset[5000 + 6]

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=2, pretrained_backbone=True, box_score_thresh=0.05)

    model.load_state_dict(torch.load("D:\LP\checkpoint\checkpoint_mobi3_0"))
    model.eval()
    pre = model([img])
    pre = pre[0]
    print(pre)
    print(boxes)
    tmp_image = (img * 255).type(dtype=torch.uint8)
    tmp_image = torchvision.utils.draw_bounding_boxes(tmp_image, boxes=pre['boxes'][:1, :], colors="red")
    tmp_image = torchvision.transforms.ToPILImage()(tmp_image)
    tmp_image.show()

    

