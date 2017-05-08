# -*- coding: utf-8 -*-
"""

Created on 2017/4/13

@author: ybwang
"""

import time, threading
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

# global parameters
index = 0
mutex = threading.Lock()
result = []
name = []
regularization = 0

# buildin parameters
iter = 1  # repeat number
threadNum = 1
cv = 10  # nfold cross validation to select Cs


def train(datafile, Cs=50, input_C=-1):
    '''    
    :param datafile: data.txt or data2.txt
    :return: 
    '''
    data = readTSV(datafile)
    x, y = data[:, 0:-1], data[:, -1]
    print 'Train file:', datafile, '\tX shape:', x.shape
    thread_list = []
    for i in xrange(threadNum):
        sthread = threading.Thread(target=run, args=(str(i), x, y, Cs, input_C))
        sthread.setDaemon(True)
        sthread.start()
        thread_list.append(sthread)
    for i in xrange(threadNum):
        thread_list[i].join()
        # print "Main thread"


def get_train_result():
    return result, regularization


def reset():
    global index, result, name, regularization
    index, result, name, regularization = 0, [], [], 0


def run(threadIndex, x, y, Cs=50, input_C=-1):
    global index, result, regularization, name
    num = 0
    while 1:
        if mutex.acquire(1):
            if index == iter:
                # print "Thread-%s: finish!" % threadIndex
                mutex.release()
                break
            print "Thread-%s: acquired %s" % (threadIndex, index)
            num = index
            index += 1
            mutex.release()
        # start do something with 'num'
        # cv to get best regularization parameter
        if input_C > 0:
            regularization = input_C
        else:
            lr = LogisticRegressionCV(Cs=Cs, penalty='l2', cv=cv, solver='newton-cg', n_jobs=1,
                                      scoring='roc_auc',
                                      refit=True)
            lr.fit(x, y)
            regularization = lr.C_[0]
        print 'regularization parameter:', regularization, '\tscore:', lr.score(x, y)
        # refit using all data and C
        lrall = LogisticRegression(penalty='l2', C=regularization, solver='newton-cg', n_jobs=1)
        lrall.fit(x, y)
        coef = lrall.coef_.ravel()
        result.append(lr.intercept_.tolist() + coef.tolist())

        # end


def readTSV(tsvfile, skipcolumn=True, skiprow=True):
    data = []
    n = 0
    with open(tsvfile, 'r') as f:
        for line in f:
            if n == 0 and skiprow == True:
                n += 1
                continue
            ele = line.rstrip().split('\t')
            if skipcolumn == True: ele = ele[1::]
            data.append(ele)
    return np.float64(np.array(data))


if __name__ == '__main__':
    train('hp/dataPLS.txt')
    r, n = get_train_result()
    print len(r[0])
