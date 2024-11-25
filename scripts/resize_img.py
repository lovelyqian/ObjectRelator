from PIL import Image
import cv2
import numpy as np

path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/92b2221b-ae92-44f0-bb31-e2d27cb736d6/aria01_214-1/0.jpg"
save = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/piano_test/aria01_214-1/0.jpg"

#cv2
# img = Image.open(path)
# img = np.array(img)
# new_img = cv2.resize(img, (960,540))
# cv2.imwrite(save,new_img)

#PIL
img = Image.open(path)
new_size = (960, 540)
img_resized = img.resize(new_size)
img_resized.save(save)

