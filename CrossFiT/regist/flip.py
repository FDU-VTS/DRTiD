from PIL import Image
import random

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img1, img2, img1_op, img2_op):
        if random.random() < self.p:
            img1_opx = 1-img1_op[0]
            img2_opx = 1-img2_op[0]
            return (img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT),(img1_opx,img1_op[1]),(img2_opx,img2_op[1]))
        return img1, img2, img1_op, img2_op


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img1, img2, img1_op, img2_op):
        if random.random() < self.p:
            img1_opy = 1-img1_op[1]
            img2_opy = 1-img2_op[1]
            return (img1.transpose(Image.FLIP_TOP_BOTTOM), img2.transpose(Image.FLIP_TOP_BOTTOM),(img1_op[0],img1_opy),(img2_op[0],img2_opy))
        return img1, img2, img1_op, img2_op


class RandomHorizontallyFlip_img(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            return img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2


class RandomVerticallyFlip_img(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            return img1.transpose(Image.FLIP_TOP_BOTTOM), img2.transpose(Image.FLIP_TOP_BOTTOM)
        return img1, img2