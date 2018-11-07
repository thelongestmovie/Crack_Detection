from PIL import Image
from PIL import ImageFilter
import cv2
import os
import numpy as np

# def find_contours(file):
#     for jpg in os.listdir(file):
#         jpg_file = os.path.join(file,jpg)
#         img = cv2.imread(jpg_file)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     #_,contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#         _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         if contours is None:
#             print('fuck!!!!')
#     # print(contours)
#     # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
#     # cv2.imwrite('/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/edge_2.jpg',img)


def main():
    im = Image.open("/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/G314A-1036+541000-1036+711840.jpg")
    #
    im_edge = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # im_contour = im.filter(ImageFilter.CONTOUR)
    # im_sharp = im.filter(ImageFilter.SHARPEN)
    #

    img_1 = cv2.imread(
        "/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/G314A-1036+541000-1036+711840.jpg")
    print("fuck!!!!")
    print img_1

    img = np.array(im)
    print (img.shape)
    #img = img.reshape(img_1.shape)

    #print img

    if img == img_1:
        print("cao!!!!")
    #
    # im_contour.save('/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/contour.jpg','jpeg')
    # im_edge.save('/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/edge.jpg', 'jpeg')
    # im_sharp.save('/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/sharp.jpg', 'jpeg')
    #img = cv2.imread('/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg/edge.jpg')
    # file = '/home/yangyuhao/data/road/data/1_data/149'
    # find_contours(file)

if __name__ == '__main__':
    main()