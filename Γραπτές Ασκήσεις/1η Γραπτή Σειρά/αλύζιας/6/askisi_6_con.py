import sys
from collections import defaultdict
orig_stdout=sys.stdout
my_dic={}
input_file=open('seira.txt','r')
out=open('apodoxes.txt','w')
sys.stdout=out
line=input_file.readlines()
teliko=1
for i in range(0,len(line)):
	word=line[i].split()
	arxiko=0
	for i in range(0,len(word)):
		print(arxiko,teliko,word[i],word[i])
		arxiko=teliko
		teliko=teliko+1
print(teliko-1)		

sys.stdout=orig_stdout
out.close()
input_file.close()


input_file = open('filr.txt','r')

line=input_file.readlines()

for i in range(0,len(line)):
	word=line[i].split()
	for j in range(0,len(line)):
		word1=line[j].split()
		my_dic[(word[0],word1[0])]=abs(len(word[0])-len(word1[0]))


out=open('bigram.txt','w')
sys.stdout=out
final=[]
arxiki=1
teliki=arxiki+1
for key in my_dic:
	print(0,arxiki,key[0],key[0],0)
	print(arxiki,teliki,key[1],key[1],my_dic[key])
	final.append(teliki)
	arxiki=teliki+1
	teliki=teliki+1

for i in range(0,len(final)):
	print(final[i])

sys.stdout=orig_stdout
out.close()
input_file.close()
