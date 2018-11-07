import cv2
import os
import sys
sys.path.append('/home/yangyuhao/data/road/postprocess')
print (sys.path)
from draw_box_1 import Draw
path = '/home/yangyuhao/data/road/data/1'
save_path = '/home/yangyuhao/data/road/preprocess/label_filter/label_1.txt'


# Bad_BlockPos:21,7
global img
global point1, point2
# def on_mouse(event, x, y, flags, param):
#     global img, point1, point2
#     if event == cv2.EVENT_LBUTTONDOWN:
#         #point1 = (x/100,y/100)
#         print (x,y)
#         n_x = x / 100
#         n_y = y / 100
#         param[0].write('Bad_BlockPos:' + str(n_y) + ',' + str(n_x))
#         param[0].write('\n')
#     # if event == cv2.EVENT_RBUTTONDBLCLK:
#     #     return
#
#     # elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
#     #     cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
#     #     cv2.imshow('image', img2)
#     # elif event == cv2.EVENT_LBUTTONUP:
#     #     point2 = (x,y)
#     #     cv2.rectangle(img2, point1, point2, (0,0,255), 5)
#     #     cv2.imshow('image', img2)
#     #     min_x = min(point1[0],point2[0])
#     #     min_y = min(point1[1],point2[1])
#     #     width = abs(point1[0] - point2[0])
#     #     height = abs(point1[1] -point2[1])


def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    if event == cv2.EVENT_LBUTTONDOWN:
        #point1 = (x/100,y/100)
        print (x,y)
        # n_x = x / 100
        # n_y = y / 100
        param[1] = param[1] + 1
        print (param[1])
        param[0].write('Bad_BlockPos:' + str(y) + ',' + str(x))
        param[0].write('\n')


def main():
    global img, count
    count = 0
    flag = True
    l = os.listdir(path)
    idx = 7000
    if os.path.exists(save_path):
        os.remove(save_path)
    f = open(save_path, 'w')
    while idx < len(l):
        file = os.path.join(path,l[idx])
        if file.split('.')[-1] == 'txt':
            idx = idx +1
            continue
        if flag:
            draw_label = Draw(file)
            img = draw_label.drawbox_label()

            f.write(str(file) + ':')
            f.write('\n')
            print file
        # if img.shape() == None:
        #     print ('fuck!!!')
        cv2.namedWindow("[image]", 0)
        cv2.setMouseCallback('[image]', on_mouse, [f,count])
        cv2.resizeWindow("[image]", 1512, 1024)
        cv2.moveWindow("[image]", 0, 0)
        cv2.imshow('[image]', img)
        waitkey_num = cv2.waitKeyEx()
        # print waitkey_num
        # if waitkey_num == 65362:
        #     print("up")
        # if waitkey_num == 65364:
        #     print("down")
        if waitkey_num == 65361:
            print("left image")
            flag = True
            idx = idx-1
        elif waitkey_num == 65363:
            print("right image")
            flag = True
            idx = idx + 1
        elif waitkey_num == 113:
            print("Exit")
            break
        else:
            flag = False
            print('Fresh!')
        cv2.destroyAllWindows()
    f.close()


if __name__ == '__main__':
    main()
