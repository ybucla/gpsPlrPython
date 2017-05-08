# -*- coding: utf-8 -*-
"""

Created on 2017/4/13

@author: ybwang
"""

import numpy as np


def main():
    kinase, code = 'AGC_PKC', 'ST'
    # kinase = kinase.replace('/', '_')

    plist = readPeptide(kinase + '/PositivePeptide')
    pls_weight = readweight(kinase + '/outpls.txt')  # no intercept
    mm_weight = readweight(kinase + '/outmm.txt')  # 1th is intercept

    gp = GpsPredictor(plist, pls_weight, mm_weight)

    randpep_list = readPeptide('random_peptide/3-ranpep_ST', False) if code == 'ST' else readPeptide(
        'random_peptide/3-ranpep_Y')
    cutoffs = gp.getcutoff(randpep_list)

    test_plist = readPeptide(kinase + '/PositivePeptide')
    test_nlist = readPeptide(kinase + '/NegativePeptide')
    test_pd = [gp.predict(p) for p in test_plist]
    test_nd = [gp.predict(p) for p in test_nlist]

    for c in cutoffs:
        tp = np.sum(np.array(test_pd) >= c)
        fn = len(test_pd) - tp
        tn = np.sum(np.array(test_nd) < c)
        fp = len(test_nd) - tn
        pr = tp / float(tp + fp)
        print 'Cutoff =', c, 'Sn =', tp / float(len(test_pd)), 'Sp =', tn / float(len(test_nd)), 'Pr =', pr


def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight


def readPeptide(pepfile, rm_duplicate=True):
    data, check = [], {}
    with open(pepfile, 'r') as f:
        for line in f:
            pep = line.rstrip()
            if rm_duplicate:
                if pep not in check:
                    data.append(pep)
            else:
                data.append(pep)
            check[pep] = ''
    return data


class GpsPredictor(object):
    def __init__(self, plist, pls_weight, mm_weight):
        '''
        initial GPS predictor using positive training set, pls_weight vector and mm_weight vector
        :param plist: (list) positive peptides list
        :param pls_weight:  (list) pls_weight vector
        :param mm_weight:   (list) mm_weight vector
        '''
        self.alist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V', 'B', 'Z', 'X', '*']
        self.plist = plist
        self.pls_weight = np.array(pls_weight).flatten()
        self.mm_weight = np.array(mm_weight).flatten()

        self.__count_matrix = self._plist_index()
        self.__mm_matrix, self.__mm_intercept = self._mmweight2matrix()

    def predict(self, query_peptide, loo=False):
        '''
        return the gps score for the query peptide
        :param query_peptide: (str) query peptide 
        :param loo: (bool) if true, count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: gps score
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(self.__count_matrix)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        pls_count_matrix = (count_clone.T * self.pls_weight).T / (len(self.plist) - rm_num)
        return np.sum(matrix * pls_count_matrix) + self.__mm_intercept

    def generatePLSdata(self, query_peptide, loo=False):
        '''
        generate the pls vector of query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, the count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: (np.ndarray) the vector of feature for each position
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        count_clone = (count_clone.T * self.pls_weight).T
        return np.sum(matrix * count_clone / (len(self.plist) - rm_num), 1)

    def generateMMdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)
        indicator_matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            indicator_matrix[i, self.alist.index(a)] = 1
        rm_num = 1 if loo else 0
        count_clone /= (len(self.plist) - rm_num)
        pls_count_matrix = (count_clone.T * self.pls_weight).T
        m = np.dot(indicator_matrix.T, pls_count_matrix) * self.__mm_matrix
        m += m.T
        np.fill_diagonal(m, np.diag(m) / float(2))
        iu1 = np.triu_indices(m.shape[0])
        return m[iu1]

    def getcutoff(self, randompeplist, sp=[0.98, 0.95, 0.85]):
        '''
        return cutoffs using 10000 random peptides as negative
        :param randompeplist: (list) random generated peptides
        :param sp: (float list) sp to be used for cutoff setting
        :return: (float list) cutoffs, same lens with sp 
        '''
        rand_scores = sorted([self.predict(p) for p in randompeplist])
        cutoffs = np.zeros(len(sp))
        for i, s in enumerate(sp):
            index = np.floor(len(rand_scores) * s).astype(int)
            cutoffs[i] = rand_scores[index]
        return cutoffs

    def _plist_index(self):
        '''
        return the amino acid frequency on each position, row: position, column: self.alist, 61 x 24
        :return: count matrix
        '''
        n, m = len(self.plist[0]), len(self.alist)
        count_matrix = np.zeros((n, m))
        for i in range(n):
            for p in self.plist:
                count_matrix[i][self.alist.index(p[i])] += 1
        return count_matrix / float(len(self.plist))

    def _mmweight2matrix(self):
        '''
        convert matrix weight vector to similarity matrix, 24 x 24, index order is self.alist
        :return: 
        '''
        aalist = self.getaalist()
        mm_matrix = np.zeros((len(self.alist), len(self.alist)))
        for n, d in enumerate(aalist):
            value = self.mm_weight[n + 1]  # mm weight contain intercept
            i, j = self.alist.index(d[0]), self.alist.index(d[1])
            mm_matrix[i, j] = value
            mm_matrix[j, i] = value
        return mm_matrix, self.mm_weight[0]

    def getMM_matrix(self):
        return self.__mm_matrix

    def getaalist(self):
        '''return aa-aa list
        AA: 0
        AR: 1
        '''
        aa = [self.alist[i] + self.alist[j] for i in range(len(self.alist)) for j in range(i, len(self.alist))]
        return aa


if __name__ == '__main__':
    main()
