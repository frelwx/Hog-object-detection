import cv2
from skimage.feature import hog
import torch
from classifier import linear_classfier
import os
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    pathes = []
    for y in range(0, image.shape[0] - window_size[0], step_size[0]):
        for x in range(0, image.shape[1] - window_size[1], step_size[1]):
            item = (y, x, image[y:y + window_size[0], x:x + window_size[1]])
            pathes.append(item)
    return pathes
img_path = "/home/lwx/HOG/data/csgo225labeled/images/img_1009.jpg"
img = cv2.imread(img_path)
print(img.shape)




checkpoint_path = "/home/lwx/HOG/new_best.pt"
model = linear_classfier()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# print(len(tmp), tmp[len(tmp) - 1])
# cv2.imwrite("./tmp.jpg", tmp[len(tmp) - 1][2])
# for item in tmp:
def recognition_one_crop(img):
    fd = hog(img, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), channel_axis=-1)
    fd = torch.from_numpy(fd)
    fd = fd.to(dtype=torch.float32)
    fd = fd.unsqueeze(dim=0)
    predict = model(fd).squeeze()
    predict = torch.nn.functional.softmax(predict, dim=-1)  
    return predict[0].item()
def recognition_all(img):
    rois = sliding_window(img, (112, 56), (20, 10))
    possibilitys = []
    print(len(rois))
    for idx in range(0, len(rois)):
        pair = (recognition_one_crop(rois[idx][2]), idx)
        possibilitys.append(pair)
    possibilitys.sort(reverse=True)
    os.system("rm -r ./predict_example")
    os.system("mkdir predict_example")
    for idx in range(0, len(possibilitys)):
        # print(possibilitys[idx])
        # tmp = "./predict_example/" + str(possibilitys[idx][0]) + "_" +  str(rois[1]) + "_" + str(rois[0]) + ".jpg"
        cv2.imwrite("./predict_example/" + str(possibilitys[idx][0]) + str(rois[idx][1] + 28) + "_" + str(rois[idx][0] + 56) + "_" + str(possibilitys[idx][0]) + ".jpg", rois[idx][2])
        # if(idx > 10):
        #     break
# print(recognition_one_crop(cv2.imread("/home/lwx/HOG/data/csgo225labeled/postives/p119.jpg")))
# recognition_all(img)
print(recognition_one_crop(img[416:528, 542:598]))