import cv2
import numpy as np


def prepare_texture(y_exp, in_files, out_file, crop_x=None):
    images = [cv2.imread(file) for file in in_files]

    list_images = []
    for image in images:
        y, x, _ = image.shape
        #print("old ", image.shape)
        x_new = int(y_exp * float(x)/y)
        image = cv2.resize(image, (x_new, y_exp))
        #print("new ",image.shape)
        list_images.append(image)
    texture = np.concatenate(list_images, axis=1)
    if crop_x is not None:
        texture = texture[:, :crop_x, :]
    texture = np.flip(texture, axis=1)
    print("texture-shape: ", texture.shape)
    cv2.imwrite(out_file, texture)

y_exp = 500
in_files_1 = [
          "./image/thuymac-1.jpeg",
          "./image/thuymac-2.jpeg",
          "./image/thuymac-3.jpeg"
          ]
out_file_1 = "./image/thuymac.jpeg"


in_files_2 = [
          "./image/lotu.jpeg",
          "./image/tieuvi.jpeg",
          "./image/ledinh.jpeg",
          "./image/thuylinh.jpeg"
          ]
out_file_2 = "./image/beauty.jpeg"
prepare_texture(y_exp, in_files_1, out_file_1, crop_x=2960)
prepare_texture(y_exp, in_files_2, out_file_2)
