from collections import defaultdict
import re

## trans pep 2 input vector
def pep2data(query,plist,aalist,weight,positive=False):
	aadict = defaultdict(float)
	aaweight = defaultdict(float)
	N = 0
	for i in xrange(0,len(plist)):
		if positive and plist[i] == query:
			continue
		for j in xrange(0,len(query)):
			aa1 = query[j] + plist[i][j]
			aa2 = plist[i][j] + query[j]
			if aa1 in aalist:
				aadict[aa1] += 1 * weight[j]
				# aaweight[aa1] += weight[j]
			else:
				aadict[aa2] += 1 * weight[j]
				# aaweight[aa2] += weight[j]
		N += 1
	d = []
	for i in xrange(0,len(aalist)):		
		if aalist[i] in aadict:
			#d.append(aadict[aalist[i]] / float(N))
			d.append(aadict[aalist[i]])
		else:
			d.append(0)
	return d

def pep2PLSdata(query,plist,blosumMatrix,positive=False):
	aadict = defaultdict(int)
	N = 0
	for i in xrange(0,len(plist)):
		if positive and plist[i] == query:
			continue
		for j in xrange(0,len(query)):
			aadict[j] += blosumMatrix[query[j]+plist[i][j]]
		N += 1
	d = []
	for i in sorted(aadict.keys()):		
		d.append(aadict[i] / float(N))
	return d

def getBlosumMatrix(matrixfile):
	aalist = []
	matrix = defaultdict(int)
	with open(matrixfile,'r') as f:
		for line in f:
			line = line.rstrip()
			if line.find('#') != -1: continue
			if re.match(re.compile('^\s+'),line):
				line = re.sub(re.compile('^\s+'),'',line)
				aalist = re.split(re.compile('\s+'),line)
			else:
				ele = re.split(re.compile('\s+'),line)
				for i in xrange(1,len(ele)):
					matrix[ele[0] + aalist[i-1]] = float(ele[i])
	return matrix

def getWeight(weightfile):
	weight = []
	with open(weightfile,'r') as f:
		for line in f:
			weight.append(float(line))
	weight.append(weight[0])	
	return weight[1:]

def getAAIndex():
	aalist = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
	aa = [aalist[i] + aalist[j] for i in range(len(aalist)) for j in range(i,len(aalist))]
	return aa

def readFileIn(pepfile):
	peplist = []
	with open(pepfile,'r') as f:
		for line in f:
			peplist.append(line.rstrip())
	return peplist

if __name__ == '__main__':
	aa = getAAIndex()
	plist = readFileIn('PositivePeptide')
	nlist = readFileIn('NegativePeptide')

	# weight = getWeight('weight.txt')
	# print weight
	# name = plist + nlist
	# v = []
	# for i in range(len(plist)):
	# 	d = pep2data(plist[i],plist,aa,weight,positive=True)
	# 	d.append(1)
	# 	v.append(d)
	# for i in range(len(nlist)):
	# 	d = pep2data(nlist[i],plist,aa,weight,positive=False)
	# 	d.append(-1)
	# 	v.append(d)
	
	# with open('data.txt','w') as fout:
	# 	fout.write("NAME\t"+"\t".join(aa)+"\tLabel\n")
	# 	for i in range(len(v)):
	# 		fout.write(name[i]+"\t")
	# 		fout.write("\t".join([str(x) for x in v[i]])+"\n")

	matrix = getBlosumMatrix('../BLOSUM62R.matrix')
	name = plist + nlist
	v = []
	for i in range(len(plist)):
		d = pep2PLSdata(plist[i],plist,matrix,positive=True)
		d.append(1)
		v.append(d)
	for i in range(len(nlist)):
		d = pep2PLSdata(nlist[i],plist,matrix,positive=False)
		d.append(-1)
		v.append(d)
	
	with open('data2.txt','w') as fout:
		fout.write("NAME\t"+"\t".join([str(x) for x in range(61)])+"\tLabel\n")
		for i in range(len(v)):
			fout.write(name[i]+"\t")
			fout.write("\t".join([str(x) for x in v[i]])+"\n")
