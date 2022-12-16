import cv2
from skimage.feature import hog
img = cv2.imread("/home/lwx/HOG/data/csgo225labeled/images/img_272.jpg")
fd, tmp = hog(img, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), channel_axis=-1, visualize=True)
cv2.imwrite("./tmp.jpg", tmp)