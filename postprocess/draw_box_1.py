import matplotlib.pyplot as plt
import cv2
import os
import argparse

# FLAGS = flags.FLAGS
#
# flags.DEFINE_string(
#     'data_dir', '/home/yangyuhao/data/road/data/test_data/G314A-1036+541000-1036+711840.jpg', 'Test image directory.')

parser = argparse.ArgumentParser(description='sb')
parser.add_argument('--data_dir', type=str, default = '/home/yangyuhao/data/road/data/test_data/G314A-1058+991800-1059+161600.jpg')
args = parser.parse_args()

class Draw:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.test_text = os.path.join(data_dir,'test.txt')
        self.img_txt = os.path.join(data_dir,data_dir.split('/')[-1].split('.')[0] + '.txt')
        self.train_txt = data_dir.split('.')[0] + '.txt'
    def drawbox(self):
        image = cv2.imread(os.path.join(self.data_dir, self.data_dir.split('/')[-1]))
        with open(self.img_txt,'r') as file:
            lines = file.readlines()
            for line in lines:
                print (line)
                if 'Bad_BlockPos' in line:
                    xyoff = line.split(':')[-1]
                    x = 100*(int(xyoff.split(',')[0])-1)
                    y = 100*(int(xyoff.split(',')[1])-1)
                    image = cv2.rectangle(image, (y, x), (y+100, x+100), (255, 0, 0), 5)
            cv2.imwrite(os.path.join(self.data_dir, 'train.jpg'), image)

    def drawbox_label(self):
        image = cv2.imread(self.data_dir)
        with open(self.train_txt,'r') as file:
            lines = file.readlines()
            for line in lines:
                #print (line)
                if 'Bad_BlockPos' in line:
                    xyoff = line.split(':')[-1]
                    x = 100*(int(xyoff.split(',')[0])-1)
                    y = 100*(int(xyoff.split(',')[1])-1)
                    image = cv2.rectangle(image, (y, x), (y+100, x+100), (255, 0, 0), 5)
        return image



    def drawbox_test(self):
        image = cv2.imread(os.path.join(self.data_dir, self.data_dir.split('/')[-1]))
        s = set([])
        with open(self.test_text,'r') as file:
            lines = file.readlines()
            for line in lines:
                type = line.split(":")[-1]
                if '1' in type:
                    continue
                if '0' in type:
                    # count = count + 1
                    xyoff = line.split(':')[0].split('.')[0]
                    x = int(xyoff.split('_')[1])
                    y = int(xyoff.split('_')[2])
                    s.add((x,y))
                    # image = cv2.rectangle(image, (y, x), (y+50, x+50), (255, 0, 0), 5)
        direct = ((-50,-50),(-50,0),(-50,50),(0,50),(0,-50),(50,-50),(50,0),(50,50))
        count = 0
        for e in s:
            flag = False
            for l in direct:
                nx = e[0] + l[0]
                ny = e[1] + l[1]
                print (nx,ny)
                if (nx,ny) in s:
                    flag = True
                    break
            if(flag):
                print(l[1],l[0])
                image = cv2.rectangle(image, (e[1], e[0]), (e[1] + 50, e[0] + 50), (255, 0, 0), 5)
                count = count + 1
        print count
        cv2.imwrite(os.path.join(self.data_dir,'test.jpg'), image)


def main():
    print(args.data_dir)
    data_dir = args.data_dir
    if '.jpg' not in data_dir:
        for jpg_dir in os.listdir(data_dir):
            file = os.path.join(data_dir,jpg_dir)
            draw = Draw(file)
            # draw.drawbox()
            draw.drawbox_test()
    else:
        draw = Draw(data_dir)
        draw.drawbox()
        draw.drawbox_test()
if __name__ == '__main__':
    main()