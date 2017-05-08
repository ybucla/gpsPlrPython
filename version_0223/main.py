# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:19:49 2017

@author: ybwang
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

def main(familyName,codeName):
    # python %prog Other/PLK/PLK1 ST
    dirName = familyName.replace('/','_')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    os.chdir(dirName)

    # 1. PLS training with multi-threading
    # generate and read data
    os.system('java -jar ../getPN/GetPosNegPeptide.jar ' + familyName + ' ' + codeName + ' ../getPN/index.txt ../getPN/new.elm')
    os.system("python ../2numd-PLS.py")
    name, data = readTSV('data2.txt')
    # train
    x,y = data[:,:-1],data[:,-1]
    result, mucoeff = train(x,y,iter=2,njobs=3)
    # write result
    writeWeight('weight_PLS.txt',name,mucoeff)
    writeCoeff('coeff_PLS.txt',name,result)

    # 2. MM training with multi-threading
    # generate and read data
    os.system("python ../2numd-MM.py")
    name, data = readTSV('data.txt')
    # train
    x,y = data[:,:-1],data[:,-1]
    result, mucoeff = train(x,y,iter=2,njobs=3)
    # write result
    writeWeight('weight_MM.txt',name,mucoeff)
    writeCoeff('coeff_MM.txt',name,result)

#    # write to roc data and plot
#    with open('rocdata.txt','w') as fout:
#        fout.write('label'+'\t'+'pls'+'\t'+'mm'+'\n')
#        for i in range(len(y)):
#            fout.write(str(y[i]) + '\t'+str(pre1[i,0])+'\t'+str(pre2[i,0])+'\n')
#    # plot roc
#    os.system('"D:/Program Files/R/R-3.3.2/bin/Rscript" ../plotroc.r')

def train(x,y,iter=200,njobs=1):
    result = [] # coef matrix, row:iter, column:features
    intercept = []
    for i in range(iter):
        print i
        lr = LogisticRegressionCV(Cs = 50, penalty='l1',cv=10, solver='liblinear',refit=True,n_jobs=njobs)
        lr.fit(x,y)
        coef = lr.coef_.ravel()
        result.append(coef.tolist())
        intercept.append(lr.intercept_)
    coef = np.row_stack(result)
    nonzerolist = (coef != 0).sum(1)
    threshold = sorted(nonzerolist,reverse=True)[int(round(len(nonzerolist[::-1]) * 0.1,0)) + 1]
    index = np.where(np.array(nonzerolist) < threshold)
    mucoeff = coef.mean(axis=0)
    mucoeff[index] = 0  # mean coeff, 1*features (sparse)
    mucoeff = np.append(mucoeff,np.mean(intercept)).tolist()
    return result, mucoeff # return two lists

def readTSV(tsvfile,skipcolumn=True,skiprow=True):
    head,data = [],[]
    n = 0
    with open(tsvfile,'r') as f:
        for line in f:
            if n == 0 and skiprow == True:
                head = line.split('\t')[1:-1]
                n += 1
                continue
            ele = line.rstrip().split('\t')
            if skipcolumn == True: ele = ele[1::]
            data.append(ele)
    return (head, np.float64(np.array(data)))

def writeWeight(weightfile,namedata,coefdata):
    with open(weightfile,'w') as fout:
        for i in range(len(namedata)):
            fout.write( namedata[i]+'\t'+str(coefdata[i])+'\n')

def writeCoeff(coefffile,namedata,coefdata):
    with open(coefffile,'w') as fout:
        fout.write('\t'.join(namedata)+'\n')
        for i in range(len(coefdata)):
            fout.write('\t'.join([str(x) for x in coefdata[i]])+'\n')

if __name__ == '__main__':
    main('AGC/DMPK/ROCK','ST')