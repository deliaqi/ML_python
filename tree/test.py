#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:46:08 2018

@author: liujiaqi
"""

#import trees,numpy
#from math import log

myDat, labels = trees.createDataSet()

initEnt = trees.calcShannonEnt(myDat)

tmp = trees.splitDataSet(myDat, 0, 1)      

bestFeature = trees.chooseBestFeatureToSplit(myDat)

myTree = trees.createTree(myDat, labels)




    
  #  print uniqueVals