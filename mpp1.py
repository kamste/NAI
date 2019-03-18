# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:56:58 2019

@author: Kamil
"""

import numpy as np
import random

def uploadXYVectors(path):
    content=list()
    y_train=list()
    with open(path) as f:
        content = f.readlines()
    for i in range (len(content)):
        content[i] = content[i].strip()
    
    content=np.array(content)
    random.shuffle(content)
    tmp = content[0].strip().split("\t")
    number_of_cols_train=len(tmp)
    number_of_rows_train=len(content)
    X_train=np.zeros(shape=(number_of_rows_train,number_of_cols_train-1))
    for i in range (len(content)):
        values = content[i].rstrip().split("\t")
        y_train.append(values[-1].strip())
        values=values[0:len(values)-1]
        for j in range(len(values)):
            X_train[i,j]=(np.float(values[j].replace(',','.')))
       
        #print(values)
    
    y_train=np.array(y_train)[np.newaxis].T
    return X_train,y_train

def euclidianDistance(row, test_row):
    distance=0
    for i in range(len(row)):
        distance=distance+(row[i]-test_row[i])*(row[i]-test_row[i])
      
    distance=np.sqrt(distance)
    return distance

#### predicting test set
    

def predictOneRow(dictionary,X_train,y_train,x):
    tmp_classes_predicted=dictionary.copy()
    distances=np.zeros(len(X_train))
    for i in range(len(X_train)):
        distances[i]=euclidianDistance(X_train[i,:],x)
    kNN_index=list()
    tmp=distances
    for i in range(k):
        a = np.argmin(tmp)
        tmp[a]
        kNN_index.append(a)
        tmp=np.delete(tmp,a)
    for i in range (len(kNN_index)):
        tmp_classes_predicted[y_train[kNN_index[i]][0]]+=1            
    predicted_class=max(tmp_classes_predicted, key=tmp_classes_predicted.get)
    return predicted_class   






        
    
def predict(classesPredicted,X_test,X_train,y_train,k):
    y_predicted=list()

    for t in range (len(X_test)):
        row=predictOneRow(classesPredicted,X_train,y_train,X_test[t,:])
        y_predicted.append(row)
        
    
    return y_predicted
 #############      

#compare two vectors
def acc(y_predicted,y_test):
    y_predicted=np.array(y_predicted)
    correct=0
    wrong=0
    for i in range (len(y_predicted)):
        if y_predicted[i]==y_test[i][0].strip() :
            correct=correct+1
        else:
            wrong=wrong+1
    return correct/(wrong+correct), correct


# cross validation
def crossValidation(classesPredicted,folds,k,X_train,y_train):
    k=int((len(X_train)/folds))
    cross_val_accuracies=list()
    prev=0
    for i in range(k):
        X_test_cross=X_train.copy()[prev:(i+1)*folds,:]
        y_test_cross=y_train[prev:(i+1)*folds,:]
        left=X_train.copy()[:prev,:]
        right=X_train.copy()[(i+1)*folds:,:]
        leftY=y_train.copy()[:prev,:]
        rightY=y_train.copy()[(i+1)*folds:,:]
        #X_train_cross=np.delete(X_train_cross,np.s_[prev:(i+1)*folds],0)
        #print('Cross validation'+str(i+1))
        X_train_cross=np.concatenate((left,right), axis=0)
        y_train_cross=np.concatenate((leftY,rightY),axis=0)
        
        y_predicted=predict(classesPredicted,X_test_cross,X_train_cross,y_train_cross,k)
        accuracy, corr=acc(y_predicted,y_test_cross)
        cross_val_accuracies.append(accuracy)
        #print(len(y_train_cross))
        prev=(i+1)*folds
    return cross_val_accuracies

#Main parameters dict,k, folds, func
#Iris-virginica ,Iris-versicolor,Iris-setosa

if __name__=='__main__':
    
    X_train,y_train=uploadXYVectors('iris_training.txt')
    X_test,y_test=uploadXYVectors('iris_test.txt')
    print (__name__)
    class_num = input('Enter number of classes: ')
    classesPredicted=dict()
    for i in range(int(class_num)):
        key = input('Enter correct class name ')
        classesPredicted[key]=0
    while(True):
        
        
        k = int(input('Enter number of nearest neighbours: ')) 
        folds =int( input('Enter crossvalidation parameter: '))
        
        y_predicted=predict(classesPredicted,X_test,X_train,y_train,k)
        accuracy, corr=acc(y_predicted,y_test)
        print('Accuracy from test set and training set: '+ str(accuracy)+': ('+str(corr)+' good pred)')
        list_of_acc=crossValidation(classesPredicted,folds,k,X_train,y_train)
        print('K-fold cross-validation accuracies: '+ str(list_of_acc))
        print('Mean: '+str(np.mean(list_of_acc)))
        
        again = input('Do you want to enter new parameters or enter vector to predict?[q-exit,v-enter vector: ')
        if again=='q':
            break
        elif again=='v':
            while(True):
                Vector=np.empty(len(X_train[0]))
                for i in range(len(X_train[0])):
                    Vector[i]=input('Podaj x'+str(i+1)+': ')
                prediction= predictOneRow(classesPredicted,X_train,y_train,Vector)
                print('Predicted class: '+str(prediction))
                decison=input('Do you want to enter another[n- no]')
                if decison=='n':
                    import sys
                    sys.exit()
                else:
                    continue
                
    
    
        else:
            continue

