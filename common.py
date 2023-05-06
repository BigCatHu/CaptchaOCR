import os
import random
import time

captcha_array=list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size = 4
from captcha.image import ImageCaptcha
if __name__ == '__main__':
    print(captcha_array)
    image =ImageCaptcha()
    for i in range(200):
        image_val = "".join(random.sample(captcha_array, 4))

        path = "./dataset/train/"
        os.makedirs(path,exist_ok=True)

        image_name = path + "{}_{}.png".format(image_val, int(time.time()))
        print(image_name)
        image.write(image_val,image_name)