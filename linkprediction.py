import sklearn
from scipy.sparse import lil_matrix
import numpy as np
import json
import sklearn
import random
from sklearn.metrics import roc_auc_score


def format_data_for_display(emb_file):
	i2e = dict()
	with open(emb_file, 'r') as r:
		line = r.readline()
		# node_id = 0
		for line in r:
			embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
			node_id = int(embeds[0])
			# embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
			i2e[node_id] = embeds[1:]
	
	# X = []
	# Y = []
	# for (id, label) in i2l_list:
	# X.append(i2e[id])
	# Y.append(label)
	return i2e


def getdata(posfile, negfile, num):
	listdata = []
	y_true = []
	tempfile1 = []
	with open(posfile, 'r') as r:
		for line in r:
			x = line.strip('\n').split()
			tempfile1.append(x)
	a = random.sample(tempfile1, num)
	for x in a:
		listdata.append([int(x[0]), int(x[1])])
		y_true.append(int(x[2]))
		
	tempfile2 = []
	with open(negfile, 'r') as r:
		for line in r:
			x = line.strip('\n').split()
			tempfile2.append(x)
	a = random.sample(tempfile2, num)
	for x in a:
		listdata.append([int(x[0]), int(x[1])])
		y_true.append(int(x[2]))
	y_true = np.array(y_true)
	return listdata, y_true

def sigmoid(x):
	ans = 1.0 / (1.0 + np.exp(-x))
	return ans

def calcul(a, b):
	return sigmoid(np.dot(a, b))

def rundata(listdata, oremb):
	y_scores = []
	for x in listdata:
		if(x[0] not in oremb):
			print("debug0")
			print(x[0])
			break
		if(x[1] not in oremb):
			print("debug1")
			print(x[1])
			break
		y_scores.append(calcul(oremb[x[0]],oremb[x[1]]))
	y_scores = np.array(y_scores)
	return y_scores

def funcauc(y_true, y_scores):
	ans = roc_auc_score(y_true, y_scores)
	return ans
	
def funcval(y_true, y_scores, val):
	total = len(y_scores)
	right = 0
	for i in range(total):
		if(y_scores[i]>=val and y_true[i] == 1):
			right = right+1
		if(y_scores[i]<val and y_true[i] == 0):
			right = right+1
	ans = round(right/total,6)
	return ans
	
def run(oremb,listdata, y_true):
	# print("run")
	y_scores = rundata(listdata, oremb)
#	for i in range(len(y_scores)):
#		print(str(i) + " " + str(y_scores[i]) + " " + str(y_true[i]))
	ans_auc = funcauc(y_true, y_scores)
	ans_val = funcval(y_true, y_scores, 0.5)
	print('ans_auc = %.4f' % (ans_auc))
	print('ans_val = %.4f' % (ans_val))
	
if __name__ == '__main__':
	oremb = format_data_for_display('./emb_amms/amms_etiv_attn_28.emb')
#	for i in range(100):
#		for j in range(100):
#			if(i==j):
#				continue
#			print(str(i) + " " + str(j) + " " + str(calcul(oremb[i], oremb[j])))
	for i in range(10):
		listdata, y_true = getdata('./Newdata/amms/pos.txt','./Newdata/amms/neg.txt', 1000)
		run(oremb, listdata, y_true)
