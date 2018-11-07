

import matplotlib.pyplot as plt
#import tensorflow as tf
import cv2
import os
def drawbox_test():
    image = cv2.imread('/home/yangyuhao/data/road/data/G314A-1070+991000-1071+161240.jpg')
    file = open("/home/yangyuhao/data/road/data/G314A-1070+991000-1071+161240.txt")
    while 1:
        lines = file.readlines()
        if not lines:
            break
        for line in lines:
            if 'Bad_RepairBlockPos' in line:
                xyoff = line.split(':')[-1]
                x = 100*(int(xyoff.split(',')[0])-1)
                y = 100*(int(xyoff.split(',')[1])-1)
                image = cv2.rectangle(image, (y, x), (y+100, x+100), (255, 0, 0), 2)
                print line
    file.close()

    cv2.imwrite("origin.jpg", image)
def cut_img_test():
    image = cv2.imread('/home/yangyuhao/data/road/data/G314A-1070+991000-1071+161240.jpg')
    file = open("/home/yangyuhao/data/road/data/G314A-1070+991000-1071+161240.txt")
    while 1:
        lines = file.readlines()
        if not lines:
            break
        for line in lines:
            if 'Bad_RepairBlockPos' in line:
                xyoff = line.split(':')[-1]
                x = 100 * (int(xyoff.split(',')[0]) - 1)
                y = 100 * (int(xyoff.split(',')[1]) - 1)
                img = image[x:x+100,y:y+100]
                cv2.imwrite(os.path.join('/home/yangyuhao/data/road/data/test',str(x)+'_'+str(y)+'.jpg'),img)
    file.close()
def img_test():
    image = cv2.imread('/home/yangyuhao/data/road/data/G314A-1070+991000-1071+161240.jpg')
    print image.shape
    image = cv2.imread('/home/yangyuhao/data/road/data/S210A-248+382640-248+353240_900_900.jpeg')
    print image.shape
    # file = open("/home/yangyuhao/data/road/data/G314A-1070+991000-1071+161240.txt")
    # while 1:
    #     lines = file.readlines()
    #     if not lines:
    #         break
    #     for line in lines:
    #         if 'Bad_RepairBlockPos' in line:
    #             xyoff = line.split(':')[-1]
    #             x = 100*(int(xyoff.split(',')[0])-1)
    #             y = 100*(int(xyoff.split(',')[1])-1)
    #             image = cv2.rectangle(image, (y, x), (y+100, x+100), (255, 0, 0), 2)
    #             print line
    # file.close()

def main():
    img_test()
if __name__ == '__main__':
    main()

