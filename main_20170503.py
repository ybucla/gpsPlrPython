# -*- coding: utf-8 -*-
"""

Created on 2017/4/13

@author: ybwang
"""

import time, os, sys, re, random, itertools
import numpy as np
from collections import defaultdict
import calltrain_python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from predict_20170503 import GpsPredictor, readweight, readPeptide


def train(familyName, codeName):
    '''
    training function
    :param familyName: 
    :param codeName: 
    :return: coefPLS, coefMM, fpr(loo), tpr(loo), auc(loo)
    '''
    global index, result, name
    # cd to family directory
    dirName = familyName.replace('/', '_')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    os.chdir(dirName)
    os.system(
        'java -jar ../getPN/GetPosNegPeptide.jar ' + familyName + ' ' + codeName + ' ../getPN/index.txt ../getPN/new.elm')
    aalist = getaalist()
    matrix = getBlosumMatrix('../BLOSUM62R.matrix')
    ini_weight = BlosumMatrix2Weight(matrix, aalist, True)
    plist = readPeptide('PositivePeptide', True)
    nlist = readPeptide('NegativePeptide', True)
    print 'Training data size:\tPos,', len(plist), '\tNeg,', len(nlist)

    # train PLS
    pd = generatePLSData(plist, plist, np.repeat(1, 61), ini_weight, loo=True, positive=True)
    nd = generatePLSData(nlist, plist, np.repeat(1, 61), ini_weight, loo=False, positive=False)
    write2data('dataPLS.txt', plist, pd, nlist, nd, [str(i) for i in range(len(plist[0]))])
    calltrain_python.train('dataPLS.txt', Cs=np.linspace(1e2, 1e4, 50))     # Cs=np.linspace(1e2, 1e4, 50)
    result, C = calltrain_python.get_train_result()
    coefPLS = getcoef(result)
    weight = coefPLS.flatten()[1:]
    writecoef('outpls.txt', coefPLS.flatten()[1:], [str(i) for i in range(len(plist[0]))])  # remove intercept
    writeC('C.txt', C)

    calltrain_python.reset()

    # train MM
    pd = generateMMData(plist, plist, weight, np.ones_like(ini_weight), loo=True, positive=True)
    nd = generateMMData(nlist, plist, weight, np.ones_like(ini_weight), loo=False, positive=False)
    write2data('dataMM.txt', plist, pd, nlist, nd, aalist)
    calltrain_python.train('dataMM.txt', Cs=np.linspace(1e-4, 500, 50))     # Cs=np.linspace(1e-4, 10, 50)
    result, C = calltrain_python.get_train_result()
    coefMM = getcoef(result)
    # coefMM = np.concatenate((np.ones(1), ini_weight)).reshape((1, -1))    # do not train mm_weight, use blosum62 instead
    writecoef('outmm.txt', coefMM, ['intercept'] + aalist)  # keep intercept
    writeC('C.txt', C)

    # plot roc
    x_p = np.hstack((np.ones((len(pd), 1)), np.row_stack(pd)))
    x_n = np.hstack((np.ones((len(nd), 1)), np.row_stack(nd)))
    x = np.vstack((x_p, x_n))
    y = x[:, -1]
    x = x[:, 0:-1]
    score = 1.0 / (1.0 + np.exp(-1 * np.dot(x, coefMM.T)))
    auc = roc_auc_score(y, score)
    fpr, tpr, threshold = roc_curve(y, score, pos_label=1)
    # plt.plot(fpr, tpr, 'r-')
    # plt.show()
    return coefPLS, coefMM, fpr, tpr, auc


def nfold(familyName, codeName, n=5):
    '''
    standard nfold cross validation, which k-1 for train and 1 for test
    '''
    # cd to family directory
    dirName = familyName.replace('/', '_')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    os.chdir(dirName)
    # os.system(
    #     'java -jar ../getPN/GetPosNegPeptide.jar ' + familyName + ' ' + codeName + ' ../getPN/index.txt ../getPN/new.elm')
    #
    aalist = getaalist()
    matrix = getBlosumMatrix('../BLOSUM62R.matrix')
    ini_weight = BlosumMatrix2Weight(matrix, aalist, True)
    plist = readPeptide('PositivePeptide', True)
    nlist = readPeptide('NegativePeptide', True)
    random.shuffle(nlist)
    nlist = nlist[0:len(plist)]

    ylabel, yscore = [], []

    if n == 0:  # loo
        pls_weight = readweight('outpls.txt')
        mm_weight = readweight('outmm.txt')
        gp = GpsPredictor(plist, pls_weight, mm_weight)
        pd = [gp.predict(p, loo=True) for p in plist]
        nd = [gp.predict(p) for p in nlist]
        yscore = pd + nd
        ylabel = np.concatenate((np.repeat(1, len(pd)), np.repeat(-1, len(nd))))
        auc = roc_auc_score(ylabel, yscore)
        fpr, tpr, threshold = roc_curve(ylabel, yscore, pos_label=1)
        os.chdir('../')
        return fpr, tpr, auc

    p_split, n_split = nfold_split(plist, nlist, n)
    for i in range(n):
        p_train = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j != i]))
        n_train = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j != i]))
        p_test = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j == i]))
        n_test = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j == i]))
        print len(p_train), len(n_train), len(p_test), len(n_test)

        # train PLS
        pd = generatePLSData(p_train, p_train, np.repeat(1, 61), ini_weight, loo=True, positive=True)
        nd = generatePLSData(n_train, p_train, np.repeat(1, 61), ini_weight, loo=False, positive=False)
        write2data(str(i) + 'fold_dataPLS.txt', p_train, pd, n_train, nd, [str(j) for j in range(61)])
        calltrain_python.train(str(i) + 'fold_dataPLS.txt', Cs=np.linspace(1e2, 1e4, 50))   # Cs=np.linspace(1e2, 1e4, 50)
        result, name = calltrain_python.get_train_result()
        coefPLS = getcoef(result)
        pls_weight = coefPLS.flatten()[1:]

        calltrain_python.reset()

        # train MM
        pd = generateMMData(p_train, p_train, pls_weight, np.ones_like(ini_weight), loo=True, positive=True)
        nd = generateMMData(n_train, p_train, pls_weight, np.ones_like(ini_weight), loo=False, positive=False)
        write2data(str(i) + 'fold_dataMM.txt', p_train, pd, n_train, nd, aalist)
        calltrain_python.train(str(i) + 'fold_dataMM.txt', Cs=np.linspace(1e-4, 10, 50))    # Cs=np.linspace(1e-4, 10, 50)
        result, name = calltrain_python.get_train_result()
        mm_weight = getcoef(result)

        calltrain_python.reset()

        # test
        gp = GpsPredictor(p_train, pls_weight, mm_weight)
        pd_test = [gp.predict(pep) for pep in p_test]
        nd_test = [gp.predict(pep) for pep in n_test]
        y = np.concatenate((np.repeat(1, len(pd_test)), np.repeat(-1, len(nd_test))))
        ylabel += y.tolist()
        yscore += pd_test + nd_test

    auc = roc_auc_score(ylabel, yscore)
    fpr, tpr, threshold = roc_curve(ylabel, yscore, pos_label=1)
    os.chdir('../')
    return fpr, tpr, auc


def nfold_gps(familyName, codeName, coefPLS, coefMM, n=5):
    '''
    nfold method which was used in GPS 2.0, the fixed model was used
    :param familyName: 
    :param codeName: 
    :param coefPLS: 
    :param coefMM: 
    :param n: 
    :return: fpr(nfold), tpr(nfold), auc(nfold)
    '''
    print 'N-fold:', n
    plist = readPeptide('PositivePeptide', True)
    nlist = readPeptide('NegativePeptide', True)
    pls_weight = coefPLS.flatten()[1:]
    mm_weight = coefMM.flatten()

    ylabel, yscore = [], []

    if n == 0:  # loo
        gp = GpsPredictor(plist, pls_weight, mm_weight)
        pd = [gp.predict(p, loo=True) for p in plist]
        nd = [gp.predict(p) for p in nlist]
        yscore = pd + nd
        ylabel = [1 for i in range(len(pd))] + [-1 for i in range(len(nd))]
        auc = roc_auc_score(ylabel, yscore)
        fpr, tpr, threshold = roc_curve(ylabel, yscore, pos_label=1)
        return fpr, tpr, auc

    repnum = 5 if len(plist) < 100 else 1

    for rn in range(repnum):
        random.shuffle(nlist)
        random.shuffle(plist)
        p_split, n_split = nfold_split(plist, nlist, n)
        for i in range(n):
            p_train = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j != i]))
            n_train = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j != i]))
            p_test = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j == i]))
            n_test = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j == i]))
            print len(p_train), len(n_train), len(p_test), len(n_test)

            gp = GpsPredictor(p_train, pls_weight, mm_weight)
            ylabel += [1 for i in range(len(p_test))] + [-1 for i in range(len(n_test))]
            yscore += [gp.predict(pep) for pep in p_test] + [gp.predict(pep) for pep in n_test]
    auc = roc_auc_score(ylabel, yscore)
    fpr, tpr, threshold = roc_curve(ylabel, yscore, pos_label=1)
    return fpr, tpr, auc


def getcoef(result):
    coef = np.row_stack(result)
    nonzerolist = []
    for i in range(coef.shape[1]):
        d = coef[:, i]
        nonzerolist.append(np.size(d.nonzero()))
    # threshold = sorted(nonzerolist, reverse=True)[int(round(len(nonzerolist[::-1]) * 0.1, 0)) + 1]
    # index = np.where(np.array(nonzerolist) < threshold)
    mucoeff = coef.mean(axis=0)
    # mucoeff[index] = 0
    mucoeffarr = np.float64(np.array([coef[0, :]]))
    return mucoeffarr


def writecoef(outfile, coef, name):
    with open(outfile, 'w') as fout:
        fout.write('\t'.join(name) + '\n')
        fout.write('\t'.join([str(x) for x in coef.flatten().tolist()]) + '\n')


def writeC(outfile, C):
    with open(outfile, 'a') as fout:
        fout.write(str(C) + '\n')


def generatePLSData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
    gp = GpsPredictor(plist, pls_weight, mm_weight)
    d = []
    label = [1] if positive else [-1]
    for query_peptide in querylist:
        d.append(gp.generatePLSdata(query_peptide, loo).tolist() + label)
    return d


def generateMMData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
    gp = GpsPredictor(plist, pls_weight, mm_weight)
    d = []
    label = [1] if positive else [-1]
    for query_peptide in querylist:
        d.append(gp.generateMMdata(query_peptide, loo).tolist() + label)
    return d


def nfold_split(plist, nlist, n=5):
    pindex = np.random.choice(range(len(plist)), len(plist), False)
    pLabel = []
    label = 1
    for i, d in enumerate(pindex):
        pLabel.append(label)
        if (i + 1) % n == 0:
            label = 1
        else:
            label += 1
    label = 1
    nLabel = []
    nindex = np.random.choice(range(len(nlist)), len(nlist), False)
    for i, d in enumerate(nindex):
        nLabel.append(label)
        if (i + 1) % n == 0:
            label = 1
        else:
            label += 1
    psplit = [[] for i in range(n)]
    for i, d in enumerate(pindex):
        psplit[pLabel[i] - 1].append(plist[d])
    nsplit = [[] for i in range(n)]
    for i, d in enumerate(nindex):
        nsplit[nLabel[i] - 1].append(nlist[d])
    return psplit, nsplit


def write2data(outfile, plist, pd, nlist, nd, head):
    '''
    write data to file for training
    :param outfile: 
    :param plist: 
    :param pd: 
    :param nlist: 
    :param nd: 
    :return: 
    '''
    with open(outfile, 'w') as fout:
        fout.write("NAME\t" + "\t".join([str(x) for x in head]) + "\tLabel\n")
        for i, p in enumerate(plist):
            fout.write(plist[i] + "\t")
            fout.write("\t".join([str(x) for x in pd[i]]) + "\n")
        for i, p in enumerate(nlist):
            fout.write(nlist[i] + "\t")
            fout.write("\t".join([str(x) for x in nd[i]]) + "\n")


def getBlosumMatrix(matrixfile):
    aalist = []
    matrix = defaultdict(int)
    with open(matrixfile, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.find('#') != -1: continue
            if re.match(re.compile('^\s+'), line):
                line = re.sub(re.compile('^\s+'), '', line)
                aalist = re.split(re.compile('\s+'), line)
            else:
                ele = re.split(re.compile('\s+'), line)
                for i in xrange(1, len(ele)):
                    matrix[ele[0] + aalist[i - 1]] = float(ele[i])
    return matrix


def BlosumMatrix2Weight(matrix, aalist, intercept=False):
    mm_weight = np.zeros(len(aalist) + 1) if intercept else np.zeros(len(aalist))
    for i, aa in enumerate(aalist):
        if intercept:
            mm_weight[i + 1] = matrix[aa]
        else:
            mm_weight[i] = matrix[aa]
    return mm_weight


def getaalist():
    '''return aa-aa list
    AA: 0
    AR: 1
    '''
    aalist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B',
              'Z', 'X', '*']
    aa = [aalist[i] + aalist[j] for i in range(len(aalist)) for j in range(i, len(aalist))]
    return aa


if __name__ == '__main__':
    # family, code = sys.argv[1], sys.argv[2]
    family, code = 'TK/Src', 'Y'

    coefPLS, coefMM, fpr, tpr, auc = train(family, code)
    fpr0, tpr0, auc0 = nfold_gps(family, code, coefPLS, coefMM, 0)
    fpr4, tpr4, auc4 = nfold_gps(family, code, coefPLS, coefMM, 4)
    fpr6, tpr6, auc6 = nfold_gps(family, code, coefPLS, coefMM, 6)
    fpr8, tpr8, auc8 = nfold_gps(family, code, coefPLS, coefMM, 8)
    fpr10, tpr10, auc10 = nfold_gps(family, code, coefPLS, coefMM, 10)

    print auc0, auc4, auc6, auc8, auc10
    label = ['Loo ' + '(AROC = ' + str(round(auc0, 4)) + ')', '4fold ' + '(AROC = ' + str(round(auc4, 4)) + ')',
             '6fold ' + '(AROC = ' + str(round(auc6, 4)) + ')',
             '8fold ' + '(AROC = ' + str(round(auc8, 4)) + ')', '10fold ' + '(AROC = ' + str(round(auc10, 4)) + ')']
    plt.plot(fpr0, tpr0, 'r-', fpr4, tpr4, 'b-', fpr6, tpr6, 'g-', fpr8, tpr8, 'c-', fpr10, tpr10, 'y')
    plt.legend(label, loc=4)
    plt.savefig('roc.png', dpi=300)


    # fpr0, tpr0, auc0 = nfold(family, code, 0)
    # fpr4, tpr4, auc4 = nfold(family, code, 4)
    # fpr6, tpr6, auc6 = nfold(family, code, 6)
    # fpr8, tpr8, auc8 = nfold(family, code, 8)
    # fpr10, tpr10, auc10 = nfold(family, code, 10)
    #
    # label = ['loo ' + '(AROC = ' + str(round(auc0, 3)) + ')',
    #          '4fold ' + '(AROC = ' + str(round(auc4, 3)) + ')',
    #          '6fold ' + '(AROC = ' + str(round(auc6, 3)) + ')',
    #          '8fold ' + '(AROC = ' + str(round(auc8, 3)) + ')', '10fold ' + '(AROC = ' + str(round(auc10, 3)) + ')']
    # plt.plot(fpr0, tpr0, 'r-', fpr4, tpr4, 'b-', fpr6, tpr6, 'g-', fpr8, tpr8, 'c-', fpr10, tpr10, 'y')
    # plt.legend(label, loc=4)
    # plt.show()
