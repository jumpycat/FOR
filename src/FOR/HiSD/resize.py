import os
import cv2
import glob



root = r'/root/autodl-tmp/data/00-DYS/CelebAMask-HQ/CelebA-HQ-img'
dst = r'/root/autodl-tmp/data/00-DYS/CelebAMask-HQ128_for_HiSD/imgs'


# print(glob.glob(f'{root}/*.jpg'))

img_names = os.listdir(root)

for img_name in img_names:
    img_pth = os.path.join(root,img_name)
    print(img_pth)
    print(1)
    new_img = cv2.resize(cv2.imread(img_pth), (128, 128))
    cv2.imwrite(os.path.join(dst,img_name), new_img)
