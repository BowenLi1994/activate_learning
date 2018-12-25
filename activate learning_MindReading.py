import scipy.io as sio
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt

def train_LR_classifier(X_train, y_train): #LR training classfiers
    LR_classifier = LogisticRegression()
    LR_classifier.fit(X_train, y_train)

    return LR_classifier


def test_LR_classifier(LR_classifier, X_test, y_test):
    accuracy = LR_classifier.score(X_test, y_test)
    return accuracy


def random_sampling(X_unlabeld, Y_unlabeld, no_samples):
    X = []
    Y = []
    rest_X= X_unlabeld
    rest_Y = Y_unlabeld
    for i in range(no_samples):
        n = rest_X.shape[0]
        n = n-1
        index_selected = rd.randint(0,n)
        X.append(rest_X[index_selected])
        Y.append(rest_Y[index_selected])
        rest_X = np.delete(rest_X,index_selected,0)
        rest_Y = np.delete(rest_Y,index_selected,0)
    return np.array(X),np.array(Y),np.array(rest_X),np.array(rest_Y)


def active_learning_random(X_train, Y_train, X_test, Y_test, unlabeld_sample_matrix, unlabeld_sample_label,
                           no_samples, round):
    accuracy_random = []
    LR_classifier = train_LR_classifier(X_train, Y_train)
    accuracy = test_LR_classifier(LR_classifier, X_test, Y_test)
    accuracy_random.append(accuracy)
    X_train_random = X_train
    Y_train_random = Y_train
    unlabeld_matrix_random = unlabeld_sample_matrix
    unlabeld_label__random = unlabeld_sample_label
    for i in range(round):
        X, Y, unlabeld_matrix_random, unlabeld_label__random = random_sampling(unlabeld_matrix_random, unlabeld_label__random, no_samples)
        X_train_random = np.concatenate((X_train_random, X))
        Y_train_random = np.concatenate((Y_train_random, Y))
        LR_classifier = train_LR_classifier(X_train_random, Y_train_random)
        accuracy = test_LR_classifier(LR_classifier, X_test, Y_test)
        accuracy_random.append(accuracy)

    return accuracy_random


def uncertainly_sapmpling(LR_classifier, X_unlabeld, Y_unlabeld,no_samples):
    probability = LR_classifier.predict_proba(X_unlabeld)
    entropy_list = []
    for pro in probability:
        entropy = 0
        for i in pro:
            entropy = entropy - i*math.log(i)
        entropy_list.append(entropy)
    index_selected = np.argpartition(entropy_list, -no_samples)[-no_samples:]
    X = []
    Y = []
    rest_X = X_unlabeld
    rest_Y = Y_unlabeld
    for i in index_selected:
        X.append(X_unlabeld[i])
        Y.append(Y_unlabeld[i])
        rest_X = np.delete(X_unlabeld,index_selected,0)
        rest_Y = np.delete(Y_unlabeld,index_selected,0)
    return np.array(X),np.array(Y),np.array(rest_X),np.array(rest_Y)

def active_learning_uncertainly(X_train, Y_train, X_test, Y_test, unlabeld_sample_matrix, unlabeld_sample_label, no_samples, round):
    accuracy_uncertainly = []
    LR_classifier = train_LR_classifier(X_train, Y_train)
    accuracy = test_LR_classifier(LR_classifier, X_test, Y_test)
    accuracy_uncertainly.append(accuracy)
    X_train_uncertainly = X_train
    Y_train_uncertainly = Y_train
    unlabeld_matrix_uncertainly = unlabeld_sample_matrix
    unlabeld_label_uncertainly = unlabeld_sample_label
    for i in range(round):
        X,Y,unlabeld_matrix_uncertainly,unlabeld_label_uncertainly = uncertainly_sapmpling(LR_classifier,unlabeld_matrix_uncertainly,unlabeld_label_uncertainly,no_samples)
        X_train_uncertainly = np.concatenate((X_train_uncertainly,X))
        Y_train_uncertainly = np.concatenate((Y_train_uncertainly,Y))
        LR_classifier = train_LR_classifier(X_train_uncertainly, Y_train_uncertainly)
        accuracy = test_LR_classifier(LR_classifier, X_test, Y_test)
        accuracy_uncertainly.append(accuracy)
    return accuracy_uncertainly

train_matrix1 = "Data for Assignment 3/MindReading/trainingMatrix_MindReading1.mat"
train_label1  = "Data for Assignment 3/MindReading/trainingLabels_MindReading_1.mat"
test_matrix1  = "Data for Assignment 3/MindReading/testingMatrix_MindReading1.mat"
test_label1   = "Data for Assignment 3/MindReading/testingLabels_MindReading1.mat"
unlabeld_sample_matrix1 = "Data for Assignment 3/MindReading/unlabeledMatrix_MindReading1.mat"
unlabeld_sample_label1 = "Data for Assignment 3/MindReading/unlabeledLabels_MindReading_1.mat"
train_matrix1 = sio.loadmat(train_matrix1)['trainingMatrix']
train_label1 = sio.loadmat(train_label1)['trainingLabels']
test_matrix1 = sio.loadmat(test_matrix1)['testingMatrix']
test_label1 = sio.loadmat(test_label1)['testingLabels']
unlabeld_sample_matrix1 = sio.loadmat(unlabeld_sample_matrix1)['unlabeledMatrix']
unlabeld_sample_label1 = sio.loadmat(unlabeld_sample_label1)['unlabeledLabels']


train_matrix2 = "Data for Assignment 3/MindReading/trainingMatrix_MindReading2.mat"
train_label2  = "Data for Assignment 3/MindReading/trainingLabels_MindReading_2.mat"
test_matrix2  = "Data for Assignment 3/MindReading/testingMatrix_MindReading2.mat"
test_label2   = "Data for Assignment 3/MindReading/testingLabels_MindReading2.mat"
unlabeld_sample_matrix2 = "Data for Assignment 3/MindReading/unlabeledMatrix_MindReading2.mat"
unlabeld_sample_label2 = "Data for Assignment 3/MindReading/unlabeledLabels_MindReading_2.mat"
train_matrix2 = sio.loadmat(train_matrix2)['trainingMatrix']
train_label2 = sio.loadmat(train_label2)['trainingLabels']
test_matrix2 = sio.loadmat(test_matrix2)['testingMatrix']
test_label2 = sio.loadmat(test_label2)['testingLabels']
unlabeld_sample_matrix2 = sio.loadmat(unlabeld_sample_matrix2)['unlabeledMatrix']
unlabeld_sample_label2 = sio.loadmat(unlabeld_sample_label2)['unlabeledLabels']

train_matrix3 = "Data for Assignment 3/MindReading/trainingMatrix_MindReading3.mat"
train_label3  = "Data for Assignment 3/MindReading/trainingLabels_MindReading_3.mat"
test_matrix3  = "Data for Assignment 3/MindReading/testingMatrix_MindReading3.mat"
test_label3   = "Data for Assignment 3/MindReading/testingLabels_MindReading3.mat"
unlabeld_sample_matrix3 = "Data for Assignment 3/MindReading/unlabeledMatrix_MindReading3.mat"
unlabeld_sample_label3 = "Data for Assignment 3/MindReading/unlabeledLabels_MindReading_3.mat"
train_matrix3 = sio.loadmat(train_matrix3)['trainingMatrix']
train_label3 = sio.loadmat(train_label3)['trainingLabels']
test_matrix3 = sio.loadmat(test_matrix3)['testingMatrix']
test_label3 = sio.loadmat(test_label3)['testingLabels']
unlabeld_sample_matrix3 = sio.loadmat(unlabeld_sample_matrix3)['unlabeledMatrix']
unlabeld_sample_label3 = sio.loadmat(unlabeld_sample_label3)['unlabeledLabels']




accuracy_random1 = active_learning_random(train_matrix1, train_label1, test_matrix1, test_label1, unlabeld_sample_matrix1, unlabeld_sample_label1, 10, 50)
accuracy_random2 = active_learning_random(train_matrix2, train_label2, test_matrix2, test_label2, unlabeld_sample_matrix2, unlabeld_sample_label2, 10, 50)
accuracy_random3 = active_learning_random(train_matrix3, train_label3, test_matrix3, test_label3, unlabeld_sample_matrix3, unlabeld_sample_label3, 10, 50)

accuracy_random=np.stack((accuracy_random1,accuracy_random2,accuracy_random3))
accuracy_random_final = np.mean(accuracy_random, axis=0)
#
# print(accuracy_random1)
# print(accuracy_random2)
# print(accuracy_random3)
# print(accuracy_random)
# print(accuracy_random_final)

accuracy_unceratainly1 = active_learning_uncertainly(train_matrix1, train_label1, test_matrix1, test_label1, unlabeld_sample_matrix1, unlabeld_sample_label1, 10, 50)
accuracy_unceratainly2 = active_learning_uncertainly(train_matrix2, train_label2, test_matrix2, test_label2, unlabeld_sample_matrix2, unlabeld_sample_label2, 10, 50)
accuracy_unceratainly3 = active_learning_uncertainly(train_matrix3, train_label3, test_matrix3, test_label3, unlabeld_sample_matrix3, unlabeld_sample_label3, 10, 50)

accuracy_unceratainly=np.stack((accuracy_unceratainly1,accuracy_unceratainly2,accuracy_unceratainly3))
accuracy_unceratainly_final = np.mean(accuracy_unceratainly, axis=0)


plt.figure()
lw = 2
plt.figure(figsize=(12, 12))
x = np.linspace(0, 50, 51)
print(x)
plt.plot(x, accuracy_random_final, color='blue',
         lw=lw, label='Accuracy in Active learning')
plt.plot(x, accuracy_unceratainly_final, color='red',
         lw=lw)
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('MindReading')
plt.legend(['Random sampling', 'Uncertainty sampling'], loc=2, fontsize=15)
plt.savefig('MindReading_accuracy.jpg')
plt.show()