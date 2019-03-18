# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:12:06 2019

@author: Kamil
"""
from mpp1 import uploadXYVectors
import numpy as np
def mapClassesAsBinary(element,singleClassName):
    if element==singleClassName:
        return 0
    else: 
        return 1
        
def into2Classes(y_test,singleClassName):
    singleClassName='Iris-setosa'
    y_test=np.array([mapClassesAsBinary(i[0],singleClassName) for i in y_test])  
    return y_test

def fit(X_train,y_train,num_epoches,alfa,treshhold):
    W=np.random.randint(10,size=len(X_train[0])+1)
    W[-1]=treshhold
    for j in range(num_epoches):
        good_pred=0  
        bad_pred=0
        for i in range(len(X_train)):
            X=np.append(X_train[i,:],-1)
            net=np.dot(X,W)
            if net>=0:
                y_pred=1
            else:
                y_pred=0
            if(y_train[i]-y_pred==0):
                good_pred+=1
            else:
                bad_pred+=1
            W=W+(y_train[i]-y_pred)*alfa*X
    print('Accuracy: '+str(good_pred*100/(good_pred+bad_pred))+'%')
    return W[-1], W[0:-1]

def predictOneRow(x,W,treshhold):
    net=np.dot(x,W)
    if net>=treshhold:
        return 1
    else:
        return 0
  
def predictSet(X_test,W,treshhold):
    y_predicted=list()
    for i in range(len(X_test)):
        y_predicted.append(predictOneRow(X_test[i,:],W,treshhold))
    return np.array(y_predicted)

def accuracy(y_pred,y_test):
    good=0
    bad=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            good+=1
        else:
            bad+=1
    return (good*100/(good+bad))
            
    
    
if __name__=='__main__': 
    X_train,y_train=uploadXYVectors('iris_training.txt')
    X_test,y_test=uploadXYVectors('iris_test.txt')
    y_train=into2Classes(y_train,'Iris-setosa')
    y_test=into2Classes(y_test,'Iris-setosa')

    
    alfa=0.01
    treshhold=15
    treshhold,W=fit(X_train,y_train,20,alfa,treshhold)
        
    
    y_pred=predictSet(X_test,W,treshhold)
    acc=accuracy(y_pred,y_test)
    print(acc)
    
    while(True):
        Vector=np.empty(len(X_train[0]))
        for i in range(len(X_train[0])):
            Vector[i]=input('Podaj x'+str(i+1)+': ')
        prediction= predictOneRow(Vector,W,treshhold)
        print('Predicted class: '+str(prediction))
        again=input('Wyjscie [q]')
        if again=='q':
            break
        else:
            continue

                


    
  
    