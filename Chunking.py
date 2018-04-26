from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing, metrics, tree, svm
from sklearn import linear_model
import numpy as np
from sklearn.naive_bayes import MultinomialNB


class Chunking:
    def __init__(self):
        self.trainDataLocation = "train.txt"
        self.testDataLocation = "test.txt"

        self.trainDataParsed = []
        self.testDataParsed = []

        self.xListTrain = []
        self.xListTest = []

        self.yListTrain = []
        self.yListTest = []

        self.XTrain = ""
        self.XTest = ""

        self.YTrain = ""
        self.YTest = ""
        self.trainDataParsed = np.genfromtxt("train.txt", dtype='str')
        self.testDataParsed = np.genfromtxt("train.txt", dtype='str')

    def prevPos_curPos(self):
        for i in range(len(self.trainDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.trainDataParsed[i][1]
            else:
                dummyDict['prevPos'] = self.trainDataParsed[i-1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
            self.xListTrain.append(dummyDict)
            self.yListTrain.append(self.trainDataParsed[i][2])

        for i in range(len(self.testDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.testDataParsed[i][1]
            else:
                dummyDict['prevPos'] = self.testDataParsed[i-1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
            self.xListTest.append(dummyDict)
            self.yListTest.append(self.testDataParsed[i][2])

        le = preprocessing.LabelEncoder()
        v = DictVectorizer(sparse=False)

        self.YTrain = le.fit_transform(self.yListTrain)
        self.YTest = le.transform(self.yListTest)
        v.fit_transform(self.xListTrain + self.xListTest)
        self.XTrain = v.transform(self.xListTrain)
        self.XTest = v.transform(self.xListTest)

    def prevPos_curPos_nextPos(self):
        for i in range(len(self.trainDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
            elif i == len(self.trainDataParsed) - 1:
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = ""
            else:
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
            self.xListTrain.append(dummyDict)
            self.yListTrain.append(self.trainDataParsed[i][2])

        for i in range(len(self.testDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i+1][1]
            elif i == len(self.testDataParsed) - 1:
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = ""
            else:
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i+1][1]
            self.xListTest.append(dummyDict)
            self.yListTest.append(self.testDataParsed[i][2])

        le = preprocessing.LabelEncoder()
        v = DictVectorizer(sparse=False)

        self.YTrain = le.fit_transform(self.yListTrain)
        self.YTest = le.transform(self.yListTest)
        v.fit_transform(self.xListTrain + self.xListTest)
        self.XTrain = v.transform(self.xListTrain)
        self.XTest = v.transform(self.xListTest)

    def prevPos_prevPos_curPos(self):
        for i in range(len(self.trainDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.trainDataParsed[i][1]
            elif i == 1:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = self.trainDataParsed[i-1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
            else:
                dummyDict['prevprevPos'] = self.trainDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
            self.xListTrain.append(dummyDict)
            self.yListTrain.append(self.trainDataParsed[i][2])

        for i in range(len(self.testDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.testDataParsed[i][1]
            elif i == 1:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
            else:
                dummyDict['prevprevPos'] = self.testDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]

            self.xListTest.append(dummyDict)
            self.yListTest.append(self.testDataParsed[i][2])

        le = preprocessing.LabelEncoder()
        v = DictVectorizer(sparse=False)

        self.YTrain = le.fit_transform(self.yListTrain)
        self.YTest = le.transform(self.yListTest)
        v.fit_transform(self.xListTrain + self.xListTest)
        self.XTrain = v.transform(self.xListTrain)
        self.XTest = v.transform(self.xListTest)

    def prevPos_prevPos_curPos_nextPos(self):
        for i in range(len(self.trainDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
            elif i == 1:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
            elif i == len(self.trainDataParsed) - 1:
                dummyDict['prevprevPos'] = self.trainDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = ""
            else:
                dummyDict['prevprevPos'] = self.trainDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
            self.xListTrain.append(dummyDict)
            self.yListTrain.append(self.trainDataParsed[i][2])

        for i in range(len(self.testDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]
            elif i == 1:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]

            elif i == len(self.testDataParsed) - 1:
                dummyDict['prevprevPos'] = self.testDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = ""
            else:
                dummyDict['prevprevPos'] = self.testDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]

            self.xListTest.append(dummyDict)
            self.yListTest.append(self.testDataParsed[i][2])

        le = preprocessing.LabelEncoder()
        v = DictVectorizer(sparse=False)

        self.YTrain = le.fit_transform(self.yListTrain)
        self.YTest = le.transform(self.yListTest)
        v.fit_transform(self.xListTrain + self.xListTest)
        self.XTrain = v.transform(self.xListTrain)
        self.XTest = v.transform(self.xListTest)

    def prevPos_prevPos_curPos_nextPos_nextPos(self):
        for i in range(len(self.trainDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = self.trainDataParsed[i + 2][1]
            elif i == 1:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = self.trainDataParsed[i + 2][1]
            elif i == len(self.trainDataParsed) - 2:
                dummyDict['prevprevPos'] = self.trainDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = ""
            elif i == len(self.trainDataParsed) - 1:
                dummyDict['prevprevPos'] = self.trainDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = ""
                dummyDict['nextnextPos'] = ""
            else:
                dummyDict['prevprevPos'] = self.trainDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.trainDataParsed[i - 1][1]
                dummyDict['curPos'] = self.trainDataParsed[i][1]
                dummyDict['nextPos'] = self.trainDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = self.trainDataParsed[i + 2][1]
            self.xListTrain.append(dummyDict)
            self.yListTrain.append(self.trainDataParsed[i][2])

        for i in range(len(self.testDataParsed)):
            dummyDict = {}
            if i == 0:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = ""
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = self.testDataParsed[i + 2][1]
            elif i == 1:
                dummyDict['prevprevPos'] = ""
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = self.testDataParsed[i + 2][1]
            elif i == len(self.testDataParsed) - 2:
                dummyDict['prevprevPos'] = self.testDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = ""

            elif i == len(self.testDataParsed) - 1:
                dummyDict['prevprevPos'] = self.testDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = ""
                dummyDict['nextnextPos'] = ""
            else:
                dummyDict['prevprevPos'] = self.testDataParsed[i - 2][1]
                dummyDict['prevPos'] = self.testDataParsed[i - 1][1]
                dummyDict['curPos'] = self.testDataParsed[i][1]
                dummyDict['nextPos'] = self.testDataParsed[i + 1][1]
                dummyDict['nextnextPos'] = self.testDataParsed[i + 2][1]

            self.xListTest.append(dummyDict)
            self.yListTest.append(self.testDataParsed[i][2])

        le = preprocessing.LabelEncoder()
        v = DictVectorizer(sparse=False)

        self.YTrain = le.fit_transform(self.yListTrain)
        self.YTest = le.transform(self.yListTest)
        v.fit_transform(self.xListTrain + self.xListTest)
        self.XTrain = v.transform(self.xListTrain)
        self.XTest = v.transform(self.xListTest)

    def logReg(self):
        reg = linear_model.LogisticRegression()
        reg.fit(self.XTrain, self.YTrain)
        predictedResult = reg.predict(self.XTest)
        print("Accuracy: ", np.mean(self.YTest == predictedResult))
        print(metrics.classification_report(self.YTest, predictedResult))

    def DT(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.XTrain, self.YTrain)
        predictedResult = clf.predict(self.XTest)
        print("Accuracy: ", np.mean(self.YTest == predictedResult))
        print(metrics.classification_report(self.YTest, predictedResult))

def main():
    while True:
        chunkingInstance = Chunking()
        featureOp = 0;
        while(int(featureOp) <= 0 or int(featureOp) > 5):
            featureOp = input(
                  "\n\t1. Pos(t-1), Pos(t)"
                  "\n\t2. Pos(t-1), Pos(t), Pos(t+1)"
                  "\n\t3. Pos(t-2), Pos(t-1), Pos(t)"
                  "\n\t4. Pos(t-2), Pos(t-1), Pos(t), Pos(t+1)"
                  "\n\t5. Pos(t-2), Pos(t-1), Pos(t), Pos(t+1), Pos(t+2)"
            "\nEnter Your Feature Option: ")
        methodOp = 0;
        while (int(methodOp) <= 0 or int(methodOp) > 2):
            methodOp = input(
                "\n\t1. Logistic Regression"
                "\n\t2. Decison Tree"
                "\nEnter Your Method Option: ")
        print("Performance Metrics for ", end="")
        if int(featureOp) == 1:
            print("Pos(t-1), Pos(t) with ", end="")
            chunkingInstance.prevPos_curPos()
        elif int(featureOp) == 2:
            print("Pos(t-1), Pos(t), Pos(t+1) with ", end="")
            chunkingInstance.prevPos_curPos_nextPos()
        elif int(featureOp) == 3:
            print("Pos(t-2), Pos(t-1), Pos(t) with ", end="")
            chunkingInstance.prevPos_prevPos_curPos()
        elif int(featureOp) == 4:
            print("Pos(t-2), Pos(t-1), Pos(t), Pos(t+1) with ", end="")
            chunkingInstance.prevPos_prevPos_curPos_nextPos()
        else:
            print("Pos(t-2), Pos(t-1), Pos(t), Pos(t+1), Pos(t+2) with ", end="")
            chunkingInstance.prevPos_prevPos_curPos_nextPos_nextPos()

        if int(methodOp) == 1:
            print("Logistic Regression Method: \nWait for Training ...")
            chunkingInstance.logReg()
        else:
            print("Decision Tree Method: \nWait for Training ...")
            chunkingInstance.DT()

if  __name__ =='__main__':main()
