import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import neighbors
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import calculate_gd as cg
import os,shutil
file_path = '/home/yangyuhao/data/road/data/test_data/label_filter'
save_path = '/home/yangyuhao/data/road/data/test_data/label_filter/a.txt'
model_path = '/home/yangyuhao/data/road/data/test_data/label_filter/RUN/model1.m'

bin_size = 8

class Train:

    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def svm_train(self):
        clf = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf.fit(self.x_train, self.y_train.ravel())
        predict_data = clf.predict(self.x_train)
        np.savetxt('tmp.txt', predict_data)
        joblib.dump(clf, model_path)
        # print (np.mean(predict_data == self.y_train))

    def mlp_train(self):
        mlp_model = MLPClassifier(solver='adam', learning_rate='constant', learning_rate_init=0.01, max_iter=500,
                                  alpha=0.01)
        mlp_model.fit(self.x_train, self.y_train.ravel())
        predict_data = mlp_model.predict(self.x_train)
        print (np.mean(predict_data == self.y_train))

    def logistic_train(self):
        lr_model = LogisticRegression(C=100, max_iter=1000)
        lr_model.fit(self.x_train, self.y_train.ravel())
        predict_data = lr_model.predict(self.x_train)
        print(np.mean(predict_data == self.y_train))

    def tree_train(self):
        tree_model = tree.DecisionTreeClassifier(criterion='gini')
        tree_model.fit(self.x_train,self.y_train.ravel())
        predict_data = tree_model.predict(self.x_train)
        print(np.mean(predict_data == self.y_train))

    def knn_train(self):
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=7)
        knn_model.fit(self.x_train, self.y_train.ravel())
        predict_data = knn_model.predict(self.x_train)
        print(np.mean(predict_data == self.y_train))


class Inference:
    def __init__(self):
        pass
    def inference(self):
        cg_infer = cg.Caculate_gd()
        file_dir = os.path.join(file_path, 'bad')
        copy_path = os.path.join(file_path, 'bad_1')
        if not os.path.exists(copy_path):
            os.mkdir(copy_path)
        clf = joblib.load(model_path)
        for file in os.listdir(file_dir):
            path = os.path.join(file_dir, file)
            l = cg_infer.caculate_each(path)
            data = np.array(l).reshape(1,-1)
            # print (clf.predict(data))
            if clf.predict(data)[0] == 1:
                print ('fuck!!')
                old_path = os.path.join(file_dir,file)
                new_path = os.path.join(copy_path,file)
                shutil.copyfile(old_path, new_path)


class Data_process:
    def __init__(self):
        pass
    def img_type(self,s):
        it = {'good':0,'bad':1}
        return it[s]

    def preprocess(self):
        cg_train = cg.Caculate_gd(bin_size, file_path, save_path)
        cg_train.caculate_train()
        data = np.loadtxt(save_path,dtype = int,delimiter = ',',converters = {8:self.img_type})
        x, y = np.split(data, (8,), axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, train_size=0.9)
        return x_train,x_test,y_train,y_test


def main():
    pd = Data_process()
    x_train, x_test, y_train, y_test = pd.preprocess()
    # print (x_train)
    # print (type(x_train))
    np.savetxt('x.txt', x_train)
    np.savetxt('y.txt', y_train)
    process = Train(x_train,x_test,y_train,y_test)
    process.svm_train()
    # process.mlp_train()
    # process.logistic_train()
    # process.tree_train()
    # process.knn_train()
    # infer = Inference()
    # infer.inference()

if __name__ == '__main__':
    main()