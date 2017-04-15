# -*- coding: utf-8 -*-
"""

Created on 2017/4/13

@author: ybwang
"""

import time, threading
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

# global parameters
index = 0
mutex = threading.Lock()
result = []
name = []

# buildin parameters
iter = 1  # repeat number
threadNum = 1
cv = 10  # nfold cross validation to select Cs


def train(datafile):
    '''    
    :param datafile: data.txt or data2.txt
    :return: 
    '''
    data = readTSV(datafile)
    x, y = data[:, 0:-1], data[:, -1]
    thread_list = []
    for i in xrange(threadNum):
        sthread = threading.Thread(target=run, args=(str(i), x, y))
        sthread.setDaemon(True)
        sthread.start()
        thread_list.append(sthread)
    for i in xrange(threadNum):
        thread_list[i].join()
        # print "Main thread"


def get_train_result():
    return result, name


def reset():
    global index, result, name
    index, result, name = 0, [], []


def run(threadIndex, x, y):
    global index, result, name
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
        lr = LogisticRegressionCV(Cs=50, penalty='l2', cv=cv, solver='newton-cg', n_jobs=1,
                                  refit=True)
        lr.fit(x, y)
        coef = lr.coef_.ravel()
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
