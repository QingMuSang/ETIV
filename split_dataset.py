import os
import random

readpath = './DBLP/'
writepath = './DBLP/'
dataname = 'dblp.txt'
labelname = 'node2label.txt'
testsetname = writepath + 'dblp_testset.txt'

def run(save_rate):
	rdataname = readpath + dataname
	rlabelname = readpath + labelname
	wdataname = writepath + dataname
	wlabelname = writepath + labelname
	
	ordata = []
	all_user = set()
	all_time = set()
	rename = dict()
	newdatasize = 0

	with open(rdataname, 'r') as r:
		for line in r:
			x = line.strip('\n').split()
			x[2] = float(x[2])
			ordata.append(x)
		ordata = sorted(ordata, key = lambda x:x[2])
		
		datasize = len(ordata)
		savesize = int(datasize * save_rate)
		print("原始数据中共有 %d 条\n预计保留 %d 条" % (datasize, savesize))

		while(savesize != datasize and ordata[savesize-1][2] == ordata[savesize][2]):
			savesize = savesize + 1
		print("实际保留 %d 条" % savesize)
		print("实际切割比例" + str(savesize/datasize))
		
		for i in range(savesize):
			x = ordata[i]
			a = str(x[0])
			b = str(x[1])
			all_user.update({a,b})
			#print(len(all_user))
			all_time.add(x[2])
		print("实际保留数据中,用户数量 %d 个,不同时间节点 %d 个" %(len(all_user), len(all_time)))
		newdatasize = savesize
		

		list_all_user = list(all_user)
		list_all_user = [int(i) for i in list_all_user]
		list_all_user.sort()
		step = 0
		for i in list_all_user:
			rename[i] = step
			#print(i, rename[i])
			step = step + 1
			
		

		flag = os.path.exists(writepath)
		if not flag:
			os.makedirs(writepath)

		with open(wdataname, 'w') as w:
			for i in range(newdatasize):
				x = ordata[i]
				a = str(rename[int(x[0])])
				b = str(rename[int(x[1])])
				w.write(a + ' ' + b + ' ' + str(x[2])+'\n')


		with open(testsetname, 'w') as w:
			index = 0
			for i in range(newdatasize,datasize):
				x = ordata[i]

				if(int(x[0]) not in rename or int(x[1]) not in rename):
					continue
				a = str(rename[int(x[0])])
				b = str(rename[int(x[1])])
				w.write(a + ' ' + b + ' ' + str(x[2])+'\n')
				index = index+1
			print('预计测试集剩余数量 %d'%(datasize-newdatasize+1))
			print('测试集剩余数量 %d'%(index))

		temp = 0
		with open(rlabelname, 'r') as r:
			with open(wlabelname, 'w') as w:
				for line in r:
					x = line.strip('\n').split()
					if(x[0] in all_user):
						temp = temp + 1
						a = str(rename[int(x[0])])
						w.write(a + ' ' + x[1] + '\n')
		print("标签集数量 " + str(temp)+ " 个")
	
if __name__ == '__main__':
	run(0.7)
