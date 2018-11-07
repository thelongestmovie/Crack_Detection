import cv2
import os
import shutil
from PIL import Image
from PIL import ImageFilter
import numpy as np

jpg_dir = '/home/yangyuhao/data/road/data/1'
good_txt_dir = '/home/yangyuhao/data/road/data/1'
txt_dir = '/home/yangyuhao/data/road/preprocess/label_filter/label_2.txt'

bad_dir = '/home/yangyuhao/data/road/data/test_data/label_filter/data_50/bad'
good_dir = '/home/yangyuhao/data/road/data/test_data/label_filter/data_50/good'

test_dir = '/home/yangyuhao/data/road/data/test_data'

#test_dir = '/home/yangyuhao/data/road/data/test_data'
# class CutImg_100:
#     def __init__(self,jpg_dir,txt_dir,repair_dir,bad_dir,good_dir):
#         self.jpg_dir = jpg_dir
#         self.txt_dir = txt_dir
#         self.repair_dir = repair_dir
#         self.bad_dir = bad_dir
#         self.good_dir = good_dir
#     def ListFile(self):
#         jpg_list = []
#         for jpg_file in os.listdir(self.jpg_dir):
#             if '.jpg' in jpg_file:
#                 txt_file = jpg_file.split('.')[0]+'.txt'
#                 jpg_list.append([jpg_file,txt_file])
#         return jpg_list
#     def main_progress_withgood(self):
#         jpg_list = self.ListFile()
#         cnt = 0
#         for file in jpg_list:
#             cnt = cnt + 1
#             if cnt >= 20:
#                 break
#             jpg_file = os.path.join(self.jpg_dir,file[0])
#             print file[0]
#             txt_file = os.path.join(self.txt_dir,file[1])
#             image = cv2.imread(jpg_file)
#             s = set([])
#             try:
#                 tmp_file = open(txt_file)
#                 while 1:
#                     lines = tmp_file.readlines()
#                     if not lines:
#                         break
#                     for line in lines:
#                         if 'Bad_BlockPos' in line:
#                             xyoff = line.split(':')[-1]
#                             x = 100 * (int(xyoff.split(',')[0]) - 1)
#                             y = 100 * (int(xyoff.split(',')[1]) - 1)
#                             s.add(str(x)+'_'+str(y))
#                             img = image[x:x + 100, y:y + 100,:]
#                             cv2.imwrite(os.path.join(self.bad_dir, file[0].split('.')[0]+'_'+str(x) + '_' + str(y) + '.jpg'), img)
#                         if 'Bad_RepairBlockPos' in line:
#                             xyoff = line.split(':')[-1]
#                             x = 100 * (int(xyoff.split(',')[0]) - 1)
#                             y = 100 * (int(xyoff.split(',')[1]) - 1)
#                             s.add(str(x) + '_' + str(y))
#                             img = image[x:x + 100, y:y + 100,:]
#                             # cv2.imwrite(os.path.join(self.repair_dir, file[0].split('.')[0]+'_'+str(x) + '_' + str(y) + '.jpg'), img)
#             except IOError:
#                 print ("Can't find file {}".format(txt_file))
#             else:
#                 tmp_file.close()
#                 for i in range(0,image.shape[0],100):
#                     for j in range(0,image.shape[1],100):
#                         tmp_str = str(i)+'_'+str(j)
#                         if tmp_str not in s:
#                             img = image[i:i + 100, j:j + 100,:]
#                             cv2.imwrite(os.path.join(self.good_dir, file[0].split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpg'), img)

# class CutImg_149_filter_col:
#     def __init__(self,jpg_dir,txt_dir,repair_dir,bad_dir,good_dir):
#         self.jpg_dir = jpg_dir
#         self.txt_dir = txt_dir
#         self.repair_dir = repair_dir
#         self.bad_dir = bad_dir
#         self.good_dir = good_dir
#     def ListFile(self):
#         jpg_list = []
#         for jpg_file in os.listdir(self.jpg_dir):
#             if '.jpg' in jpg_file:
#                 txt_file = jpg_file.split('.')[0]+'.txt'
#                 jpg_list.append([jpg_file,txt_file])
#         return jpg_list
#     def main_progress_withgood(self):
#         jpg_list = self.ListFile()
#         for file in jpg_list:
#             jpg_file = os.path.join(self.jpg_dir,file[0])
#             print file[0]
#             txt_file = os.path.join(self.txt_dir,file[1])
#             image = Image.open(jpg_file)
#             im_edge = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
#             image = np.array(im_edge)
#             s = set([])
#             try:
#                 tmp_file = open(txt_file)
#                 while 1:
#                     lines = tmp_file.readlines()
#                     if not lines:
#                         break
#                     for line in lines:
#                         if 'Bad_BlockPos' in line:
#                             xyoff = line.split(':')[-1]
#                             x = 100 * (int(xyoff.split(',')[0]) - 1)
#                             y = 100 * (int(xyoff.split(',')[1]) - 1)
#                             s.add(str(x)+'_'+str(y))
#
#                     for e in s:
#                         p = str(int(e.split('_')[0]) + 100) + '_' + e.split('_')[1]
#                         m = str(int(e.split('_')[0]) - 100) + '_' + e.split('_')[1]
#                         if m in s or p in s:
#                             x = int(e.split('_')[0])
#                             y = int(e.split('_')[1])
#                             img = image[x:x + 149, y:y + 149]
#                             cv2.imwrite(os.path.join(self.bad_dir,file[0].split('.')[0] + '_' + str(x) + '_' + str(y) + '.jpg'), img)
#
#
#             except IOError:
#                 print ("Can't find file {}".format(txt_file))
#             else:
#                 tmp_file.close()

class CutImg_test:
    def __init__(self,test_dir):
        self.test_dir = test_dir
    def ListFile(self):
        jpg_list = []
        for jpg_file in os.listdir(self.jpg_dir):
            if '.jpg' in jpg_file:
                txt_file = jpg_file.split('.')[0]+'.txt'
                jpg_list.append([jpg_file,txt_file])
        return jpg_list
    def cut_dir(self):
        jpg_list = self.ListFile()
        count = 0
        for file in jpg_list[100:]:
            if count >= 1:
                exit(0);
            count = count+1;
            #jpg_file = os.path.join(self.jpg_dir,file[0])
            jpg_file = '/home/yangyuhao/data/road/data/1/G314A-1110+834280-1111+006320.jpg'
            txt_file = '/home/yangyuhao/data/road/data/1/G314A-1110+834280-1111+006320.txt'
            print file[0]
            #txt_file = os.path.join(self.txt_dir,file[1])
            if os.path.exists(txt_file):
                image = cv2.imread(jpg_file)
                path = os.path.join(self.test_dir,file[0])
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copyfile(txt_file, os.path.join(path,file[1]))
                shutil.copyfile(jpg_file, os.path.join(path,file[0]))
                path = os.path.join(path,'jpg_data')
                if not os.path.exists(path):
                    os.makedirs(path)
                for i in range(0,image.shape[0],50):
                    for j in range(0,image.shape[1],50):
                        img = image[i:i + 50, j:j + 50,:]
                        cv2.imwrite(os.path.join(path, file[0].split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpg'), img)

    def cut_img(self):
            jpg_file = '/home/yangyuhao/data/road/data/1/G314A-1077+917040-1078+090360.jpg'
            txt_file = '/home/yangyuhao/data/road/data/1/G314A-1077+917040-1078+090360.txt'
            if os.path.exists(txt_file):
                image = cv2.imread(jpg_file)
                file = [jpg_file.split('/')[-1],txt_file.split('/')[-1]]
                path = os.path.join(self.test_dir,file[0])
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copyfile(txt_file, os.path.join(path,file[1]))
                shutil.copyfile(jpg_file, os.path.join(path,file[0]))
                path = os.path.join(path,'jpg_data')
                if not os.path.exists(path):
                    os.makedirs(path)
                for i in range(0,image.shape[0],50):
                    for j in range(0,image.shape[1],50):
                        img = image[i:i + 50, j:j + 50,:]
                        cv2.imwrite(os.path.join(path, file[0].split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpg'), img)


class CutImg_label:
    def __init__(self,txt_dir,bad_dir):
        self.txt_dir = txt_dir
        self.bad_dir = bad_dir

        self.jpg_dir = jpg_dir
        self.good_txt_dir = good_txt_dir
        self.good_dir = good_dir

    def ListFile(self):
        jpg_list = []
        for jpg_file in os.listdir(self.jpg_dir):
            if '.jpg' in jpg_file:
                txt_file = jpg_file.split('.')[0] + '.txt'
                jpg_list.append([jpg_file, txt_file])
        return jpg_list

    def cut_100(self):
        try:
            tmp_file = open(self.txt_dir)
            lines = tmp_file.readlines()
            if not lines:
                return
            idx = 0
            while idx < len(lines):
                if '/home/yangyuhao/data' in lines[idx]:
                    image = cv2.imread(lines[idx][:-2])
                    print lines[idx]
                    # print image.shape
                    file_name = lines[idx][:-2].split('/')[-1]
                    idx = idx + 1
                    while '/home/yangyuhao/data' not in lines[idx]:
                        if 'Bad_BlockPos' in lines[idx]:
                            xyoff = lines[idx].split(':')[-1]
                            x = 100 * (int(xyoff.split(',')[0]))
                            y = 100 * (int(xyoff.split(',')[1]))
                            img = image[x:x + 100, y:y + 100, :]
                            img_file = os.path.join(self.bad_dir,file_name.split('.')[0] + '_' + str(x) + '_' + str(y) + '.jpg')
                            print img_file
                            cv2.imwrite(img_file,img)
                        idx = idx + 1
                        if(idx >= len(lines)):
                            break
        except IOError:
            print ("Can't find file {}".format(self.txt_file))

    def cut_50(self):
        try:
            tmp_file = open(self.txt_dir)
            lines = tmp_file.readlines()
            if not lines:
                return
            idx = 0
            while idx < len(lines):
                if '/home/yangyuhao/data' in lines[idx]:
                    file_path = lines[idx].split(':')[0]
                    image = cv2.imread(file_path)
                    print file_path
                    # print image.shape
                    file_name = lines[idx][:-2].split('/')[-1]
                    idx = idx + 1
                    while '/home/yangyuhao/data' not in lines[idx]:
                        if 'Bad_BlockPos' in lines[idx]:
                            xyoff = lines[idx].split(':')[-1]
                            x = max(int(xyoff.split(',')[0]) - 25,0)
                            y = max(int(xyoff.split(',')[1]) - 25,0)
                            img = image[x:x + 50, y:y + 50, :]
                            img_file = os.path.join(self.bad_dir,file_name.split('.')[0] + '_' + str(x) + '_' + str(y) + '.jpg')
                            cv2.imwrite(img_file,img)
                        idx = idx + 1
                        if(idx >= len(lines)):
                            break
        except IOError:
            print ("Can't find file {}".format(self.txt_file))

    def cut_50_good(self):
        jpg_list = self.ListFile()
        cnt = 300
        for file in jpg_list:
            cnt = cnt + 1
            if cnt >= 800:
                break
            jpg_file = os.path.join(self.jpg_dir, file[0])
            print file[0]
            txt_file = os.path.join(self.good_txt_dir, file[1])
            image = cv2.imread(jpg_file)
            s = set([])
            try:
                tmp_file = open(txt_file)
                while 1:
                    lines = tmp_file.readlines()
                    if not lines:
                        break
                    for line in lines:
                        if 'Bad_BlockPos' in line:
                            xyoff = line.split(':')[-1]
                            x = 100 * (int(xyoff.split(',')[0]) - 1)
                            y = 100 * (int(xyoff.split(',')[1]) - 1)
                            s.add(str(x) + '_' + str(y))
                        # if 'Bad_RepairBlockPos' in line:
                        #     xyoff = line.split(':')[-1]
                        #     x = 100 * (int(xyoff.split(',')[0]) - 1)
                        #     y = 100 * (int(xyoff.split(',')[1]) - 1)
                        #     s.add(str(x) + '_' + str(y))
                        #     img = image[x:x + 100, y:y + 100, :]
                        #     # cv2.imwrite(os.path.join(self.repair_dir, file[0].split('.')[0]+'_'+str(x) + '_' + str(y) + '.jpg'), img)
            except IOError:
                print ("Can't find file {}".format(txt_file))
            else:
                tmp_file.close()
                for i in range(0, image.shape[0], 100):
                    for j in range(0, image.shape[1], 100):
                        tmp_str = str(i) + '_' + str(j)
                        if tmp_str not in s:
                            img = image[i:i + 50, j:j + 50, :]
                            cv2.imwrite(os.path.join(self.good_dir,
                                                     file[0].split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpg'),
                                        img)

def main():
    #cutimg = CutImg_100(jpg_dir,txt_dir,repair_dir,bad_dir,good_dir)
    # cutimg = CutImg_label(txt_dir,bad_dir)
    # cutimg.cut_50_good()
    # cutimg.cut_50()
    ci = CutImg_test(test_dir)
    ci.cut_img()
if __name__ == '__main__':
    main()
