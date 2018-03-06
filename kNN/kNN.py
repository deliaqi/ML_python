#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:31:32 2018

@author: liujiaqi
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # caculate distances
    dataSetSize = dataSet.shape[0];
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # tile(): repeat inX for dataSetSize rows and 1 column
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)  # .sum(axis=1): sum every row; axis=0: sum every column
    distances = sqDistance ** 0.5
    # select k closest points
    sortedDistIndicies = distances.argsort() # .argsort(): index number of sorted 
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # sort
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    # get number of data lines
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # get numpy
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() # delete carriage return character ('\n', '\r')
        listFromLine = line.split('\t')
        #print listFromLine
        returnMat[index,:] = listFromLine[0:3]
        #classLabelVector.append(int(listFromLine[-1]))
        classLabelVector.append(str(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] # 0: row; 1:column
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10 # select 10% as test data
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    # normalization
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs) :
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #print "the classifier came back with: %d, the real answer is:%d" % (classifierResult, datingLabels[i])
        print "the classifier came back with: %s, the real answer is:%s" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))




