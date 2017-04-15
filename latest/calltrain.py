# -*- coding: utf-8 -*-
"""

Created on 2017/4/13

@author: ybwang
"""

import time, threading
import subprocess
import numpy as np

# global parameters
index = 0
mutex = threading.Lock()
result = []
name = []

# buildin parameters
TRAIN_R_PATH = 'E:\\Pycharm\\code\\train-r.r'
iter = 1  # repeat number
threadNum = 1


def train(datafile):
    '''    
    :param datafile: data.txt or data2.txt
    :return: 
    '''
    thread_list = []
    for i in xrange(threadNum):
        sthread = threading.Thread(target=run, args=(str(i), datafile))
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


def run(threadIndex, datafile):
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
        p = subprocess.Popen('"D:/Program Files/R/R-3.3.2/bin/Rscript" ' + TRAIN_R_PATH + ' ' + datafile, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        coef = []
        tmpname = []
        tag = 0
        for line in p.stdout.readlines():
            if line.find('(Intercept)') != -1:
                tag = 1
            if tag == 0: continue
            ele = line.rstrip().split('\t')
            tmpname.append(ele[0])
            coef.append(float(ele[1]))
        result.append(coef)
        name = tmpname
        retval = p.wait()
        time.sleep(1)
        # end


if __name__ == '__main__':
    train('CMGC_CDK_CDK2_CDK2/8fold_dataPLS.txt')
    r, n = get_train_result()
    print r
