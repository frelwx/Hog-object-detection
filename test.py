import torch
import torchvision
from PIL import Image
import utils
img = Image.open("/home/lwx/HOG/data/csgo225labeled/images/img_90.jpg")
img = torchvision.transforms.ToTensor()(img)
print(img.shape)
H, W = img.shape[1:]
x = W * 0.601965
y = H * 0.56459
w = W * 0.012728
h = H * 0.042059
box = [x, y, w, h]
box = utils._box_cxcywh_to_xyxy(torch.as_tensor(box))
print(box, box.shape)

tmp_image = (img * 255).type(dtype=torch.uint8)
tmp_image = torchvision.utils.draw_bounding_boxes(tmp_image, boxes=box[None, :], colors="red")
tmp_image = torchvision.transforms.ToPILImage()(tmp_image)
tmp_image.show()
tmp_image.save("./tmp.jpg")