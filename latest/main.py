# -*- coding: utf-8 -*-
"""

Created on 2017/4/13

@author: ybwang
"""

import time, os, sys, re, random, itertools
import numpy as np
from collections import defaultdict
import calltrain, calltrain_python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def main(familyName, codeName):
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
    #
    aalist = getaalist()
    matrix = getBlosumMatrix('../BLOSUM62R.matrix')
    plist = readPeptide('PositivePeptide')
    nlist = readPeptide('NegativePeptide')
    random.shuffle(nlist)
    nlist = nlist[0:len(plist)]
    print 'Training data size:\tPos,', len(plist), '\tNeg,', len(nlist)

    # train PLS
    pd = [pep2PLSdata(p, plist, matrix, True) for p in plist]
    nd = [pep2PLSdata(p, plist, matrix, False) for p in nlist]
    write2data('dataPLS.txt', plist, pd, nlist, nd, [str(i) for i in range(61)])
    calltrain_python.train('dataPLS.txt')
    result, name = calltrain_python.get_train_result()
    coefPLS = getcoef(result)
    weight = coefPLS.flatten()[1:]
    # print 'Peptide length selection weight:\t', coefPLS.flatten()
    writecoef('outpls.txt', coefPLS.flatten()[1:], [str(i) for i in range(61)])   # remove intercept

    calltrain_python.reset()

    # train MM
    pd = [pep2MMdata(p, plist, aalist, weight, True) for p in plist]
    nd = [pep2MMdata(p, plist, aalist, weight, False) for p in nlist]
    write2data('dataMM.txt', plist, pd, nlist, nd, aalist)
    calltrain_python.train('dataMM.txt')
    result, name = calltrain_python.get_train_result()
    coefMM = getcoef(result)
    # print 'Matrix weight:\t', coefMM.flatten()
    writecoef('outmm.txt', coefMM, ['intercept']+aalist)    # keep intercept

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
    global index, result, name
    # cd to family directory
    dirName = familyName.replace('/', '_')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    os.chdir(dirName)
    os.system(
        'java -jar ../getPN/GetPosNegPeptide.jar ' + familyName + ' ' + codeName + ' ../getPN/index.txt ../getPN/new.elm')
    #
    aalist = getaalist()
    matrix = getBlosumMatrix('../BLOSUM62R.matrix')
    plist = readPeptide('PositivePeptide')
    nlist = readPeptide('NegativePeptide')
    random.shuffle(nlist)
    nlist = nlist[0:len(plist)]
    p_split, n_split = nfold_split(plist, nlist, n)

    ylabel, yscore = [], []
    for i in range(n):
        p_train = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j != i]))
        n_train = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j != i]))
        p_test = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j == i]))
        n_test = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j == i]))
        print len(p_train), len(n_train), len(p_test), len(n_test)

        # train
        pd = [pep2PLSdata(pep, p_train, matrix, True) for pep in p_train]
        nd = [pep2PLSdata(pep, p_train, matrix, False) for pep in n_train]
        write2data(str(i) + 'fold_dataPLS.txt', p_train, pd, n_train, nd, [str(j) for j in range(61)])
        calltrain_python.train(str(i) + 'fold_dataPLS.txt')
        result, name = calltrain_python.get_train_result()
        coefPLS = getcoef(result)
        weight = coefPLS.flatten()[1:]

        calltrain_python.reset()

        # train MM
        pd = [pep2MMdata(pep, p_train, aalist, weight, True) for pep in p_train]
        nd = [pep2MMdata(pep, p_train, aalist, weight, False) for pep in n_train]
        write2data(str(i) + 'fold_dataMM.txt', p_train, pd, n_train, nd, aalist)
        calltrain_python.train(str(i) + 'fold_dataMM.txt')
        result, name = calltrain_python.get_train_result()
        coefMM = getcoef(result)
        # test
        pd_test = [pep2MMdata(pep, p_train, aalist, weight, True) for pep in p_test]
        nd_test = [pep2MMdata(pep, p_train, aalist, weight, False) for pep in n_test]
        x_p = np.hstack((np.ones((len(pd_test), 1)), np.row_stack(pd_test)))
        x_n = np.hstack((np.ones((len(nd_test), 1)), np.row_stack(nd_test)))
        x = np.vstack((x_p, x_n))
        y = x[:, -1]
        x = x[:, 0:-1]
        score = 1.0 / (1.0 + np.exp(-1 * np.dot(x, coefMM.T)))
        ylabel += y.tolist()
        yscore += score.tolist()
    auc = roc_auc_score(ylabel, yscore)
    fpr, tpr, threshold = roc_curve(ylabel, yscore, pos_label=1)
    print auc
    plt.plot(fpr, tpr, 'r-')
    plt.show()


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
    aalist = getaalist()
    matrix = getBlosumMatrix('../BLOSUM62R.matrix')
    plist = readPeptide('PositivePeptide')
    nlist = readPeptide('NegativePeptide')

    ylabel, yscore = [], []

    repnum = 5 if len(plist) < 100 else 1
    for rn in range(repnum):
        random.shuffle(nlist)
        nlist = nlist[0:len(plist)]
        p_split, n_split = nfold_split(plist, nlist, n)
        for i in range(n):
            p_train = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j != i]))
            n_train = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j != i]))
            p_test = list(itertools.chain.from_iterable([x for j, x in enumerate(p_split) if j == i]))
            n_test = list(itertools.chain.from_iterable([x for j, x in enumerate(n_split) if j == i]))
            print len(p_train), len(n_train), len(p_test), len(n_test)

            weight = coefPLS.flatten()[1:]
            pd_test = [pep2MMdata(pep, p_train, aalist, weight, True) for pep in p_test]
            nd_test = [pep2MMdata(pep, p_train, aalist, weight, False) for pep in n_test]
            x_p = np.hstack((np.ones((len(pd_test), 1)), np.row_stack(pd_test)))
            x_n = np.hstack((np.ones((len(nd_test), 1)), np.row_stack(nd_test)))
            x = np.vstack((x_p, x_n))
            y = x[:, -1]
            x = x[:, 0:-1]
            score = 1.0 / (1.0 + np.exp(-1 * np.dot(x, coefMM.T)))
            ylabel += y.tolist()
            yscore += score.tolist()
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
    mucoeffarr = np.float64(np.array([coef[0,:]]))
    return mucoeffarr


def writecoef(outfile, coef, name):
    with open(outfile, 'w') as fout:
        fout.write('\t'.join(name) + '\n')
        fout.write('\t'.join([str(x) for x in coef.flatten().tolist()]) + '\n')


def pep2PLSdata(query, plist, blosumMatrix, positive=False):
    '''
    convert query peptide to data vector, weights are 1 and default matrix is BLOSUM62  
    :param query: (str) peptide
    :param plist: (list) positive peptide list
    :param blosumMatrix: (defaultdict(int)) matrix obtained from getBlosumMatrix()
    :param positive: (bool)
    :return: a numeric vector 
    '''
    aadict = defaultdict(int)
    N = 0
    for i in xrange(0, len(plist)):
        if positive and plist[i] == query:
            continue
        for j in xrange(0, len(query)):
            aadict[j] += blosumMatrix[query[j] + plist[i][j]]
        N += 1
    d = []
    for i in sorted(aadict.keys()):
        d.append(aadict[i] / float(N))
    if positive:
        d.append(1)
    else:
        d.append(0)
    return d


def pep2MMdata(query, plist, aalist, weight, positive=False):
    '''
    convert query peptide to data vector, default weights are trained coef and matrix is BLOSUM62  
    :param query: (str) peptide
    :param plist: (list) positive peptide list
    :param aalist: (list) aa list, i.e. ['AA','AR',...,'**']
    :param weight: (list) coef without 'Intercept' obtained from PLS training
    :param positive: (bool)
    :return: a numeric vector 
    '''
    aadict = defaultdict(float)
    aaweight = defaultdict(float)
    aanumdict = defaultdict(float)
    N = 0
    for i in xrange(0, len(plist)):
        if positive and plist[i] == query:
            continue
        for j in xrange(0, len(query)):
            aa1 = query[j] + plist[i][j]
            aa2 = plist[i][j] + query[j]
            if aa1 in aalist:
                aadict[aa1] += 1 * weight[j]
                # aaweight[aa1] += weight[j]
                aanumdict[aa1] += 1.0
            else:
                aadict[aa2] += 1 * weight[j]
                # aaweight[aa2] += weight[j]
                aanumdict[aa2] += 1.0
        N += 1
    d = []
    for i in xrange(0, len(aalist)):
        if aalist[i] in aadict:
            d.append(aadict[aalist[i]] / float(N))
            # d.append(aadict[aalist[i]] / float(N))
            # d.append(aadict[aalist[i]] / aanumdict[aalist[i]])
        else:
            d.append(0)
    if positive:
        d.append(1)
    else:
        d.append(0)
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


def getaalist():
    '''return aa-aa list
    AA: 0
    AR: 1
    '''
    aalist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B',
              'Z', 'X', '*']
    aa = [aalist[i] + aalist[j] for i in range(len(aalist)) for j in range(i, len(aalist))]
    return aa


def readPeptide(pepfile):
    data = []
    with open(pepfile, 'r') as f:
        for line in f:
            data.append(line.rstrip())
    return data


if __name__ == '__main__':
    # family, code = sys.argv[1], sys.argv[2]
    family, code = 'AGC/PKA', 'ST'
    coefPLS, coefMM, fpr, tpr, auc = main(family, code)
    fpr4, tpr4, auc4 = nfold_gps(family, code, coefPLS, coefMM, 4)
    fpr6, tpr6, auc6 = nfold_gps(family, code, coefPLS, coefMM, 6)
    fpr8, tpr8, auc8 = nfold_gps(family, code, coefPLS, coefMM, 8)
    fpr10, tpr10, auc10 = nfold_gps(family, code, coefPLS, coefMM, 10)

    print auc, auc4, auc6, auc8, auc10
    label = ['Loo ' + '(AROC = ' + str(round(auc, 3)) + ')', '4fold ' + '(AROC = ' + str(round(auc4, 3)) + ')',
             '6fold ' + '(AROC = ' + str(round(auc6, 3)) + ')',
             '8fold ' + '(AROC = ' + str(round(auc8, 3)) + ')', '10fold ' + '(AROC = ' + str(round(auc10, 3)) + ')']
    plt.plot(fpr, tpr, 'r-', fpr4, tpr4, 'b-', fpr6, tpr6, 'g-', fpr8, tpr8, 'c-', fpr10, tpr10, 'y')
    plt.legend(label, loc=4)
    plt.savefig('roc.png', dpi=300)
    # plt.show()
    # nfold(family, code, 10)
