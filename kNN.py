#!/usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'fish'
__createtime__ = '2015-1-26'

import numpy as np
import operator

def kNN(newInput,dataSet,lables,k):
    '''
                   newInput                                 dataset   kNN                  labels   dataset                              k   k   
    kNN               
    1.                                                               
    2.                                             
    3.                           k                  k                                                                           
    '''
    numSample = dataSet.shape[0]
    diff = np.tile(newInput,(numSample,1))-dataSet
    squredDiff = diff**2
    squredDis = np.sum(squredDiff,axis = 1)
    distance = squredDis**0.5

    sortedDistIndices = np.argsort(distance)

    classCount = {}

    for i in xrange(k):
        voteLabel = lables[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1

    maxCount = 0
    for key,value in classCount.items():
        if value>maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex

def fileMatrix(filename):
    '''
    filename:                  ,   test.txt
    fileMatrix                                                   returnMat                     classLabelVector
    '''
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = np.zeros([numberOfLines,2])
    classLabelVector = []
    index = 0

    for line in arrayLines:
        line = line.strip()
        listFormLine = line.split('\t')
        print listFormLine
        print listFormLine[0]
        returnMat[index,:] = listFormLine[0:2]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1

    return returnMat,classLabelVector

def autoNorm(dataSet):
    '''
                                                    f(x)=x-min(x)/max(x)-min(x)
                               normDataSet,      ranges,         minVals
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVal
    normDataSet = np.zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))

    return normDataSet,ranges,minVals

if __name__ == '__main__':
    '''
                         kNN                                                      
    '''
    datingDateMat,datingDateLables = fileMatrix('test.txt')
    normMat,ranges,minVals = autoNorm(datingDateMat)
    numTestVecs = int(m*normMat.shape[0])
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN(normMat[i,:],normMat[numTestVecs:m,:],datingDateLables[numTestVecs:m,:],3)
        print "kNN result : %d , real result : %d " % (classifierResult,datingDateLables[i])
        if (classifierResult != datingDateLables[i]):
            errorCount += 1.0
            print "error ratio: %f " % (errorCount/float(numTestVecs))