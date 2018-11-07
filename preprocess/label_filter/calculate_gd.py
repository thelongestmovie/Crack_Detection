import cv2
import numpy as np
import math
import os
from numpy import mean, ptp, var, std

class Caculate_gd:
    def __init__(self,bin_size = 8,file_path = None,save_path = None):
        self.bin_size = bin_size
        self.file_path = file_path
        self.save_path = save_path

    def write_txt(self,res,f,file):
        f.write(res + ',')
        f.write(file.split('/')[-2])
        f.write('\n')
    def caculate_each(self,file):
        tmp = 360 / self.bin_size
        l = [0] * self.bin_size
        # print file
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        for i in range(0, gradient_angle.shape[0], 1):
            for j in range(0, gradient_angle.shape[1], 1):
                idx = int(math.floor(gradient_angle[i][j] / tmp))
                l[idx] += 1
        # calculate variance
        data = np.array(l)
        # res = var(narray)
        res = str(ptp(data))+','+str(var(data))
        return res

    def caculate_train(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        f = open(self.save_path,'w')
        l = [os.path.join(self.file_path,'good'),os.path.join(self.file_path,'bad')]
        for file_dir in l:
            for file in os.listdir(file_dir):
                file = os.path.join(file_dir, file)
                res = self.caculate_each(file)
                self.write_txt(res,f,file)
                #print (l)
        f.close()


# file_path = '/home/yangyuhao/data/road/data/test_data/label_filter/good/G314B-100+535184-100+536410_0_1500.jpg'
# bin_size = 8
# file_path = '/home/yangyuhao/data/road/data/test_data/label_filter/good/'
# save_path = '/home/yangyuhao/data/road/data/test_data/label_filter/a.txt'
# cg = Caculate_gd(8,file_path,save_path)
# cg.caculate_sgd()

# def main():
#     caculate_sgd()
#
# if __name__ == '__main__':
#     main()