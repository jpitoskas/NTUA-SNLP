import sys
from collections import defaultdict
orig_stdout=sys.stdout
my_dic={}
plithos_dic={}
plithos_dic=defaultdict(lambda:0,plithos_dic)
lexicon_dic={}
input_file = open('lexicon.txt','r')
line = input_file.readlines()
counter=0
for i in range(0,len(line)):
	word=line[i].split()
	for j in range(0,len(word)-1):
		plithos_dic[word[j]]=plithos_dic[word[j]]+1
		if(j==len(word)-2):
			plithos_dic[word[j+1]]=plithos_dic[word[j+1]]+1
		if ((word[j],word[j+1]) in my_dic):
			if(not ((word[j]) in lexicon_dic)):
				lexicon_dic[word[j]] = word[j]
			my_dic[(word[j],word[j+1])]=my_dic[(word[j],word[j+1])]+1
			counter=counter+1
		else:
			if(not ((word[j]) in lexicon_dic)):
				lexicon_dic[word[j]] = word[j]
			my_dic[(word[j],word[j+1])] = 1
			counter += 1

out = open('lexiko.txt','w')
sys.stdout=out

for key in lexicon_dic:
	print(lexicon_dic[key])

input_file.close()
out.close()	

sys.stdout = orig_stdout

for key in my_dic:
	(a,b)=key
	my_dic[key]=my_dic[key]/float(plithos_dic[a])
	print(my_dic[key],a,b)


out = open('bigram.txt','w')
input_file=open('lexiko.txt','r')

line=input_file.readlines()

for i in range(0,len(line)):
	word = line[i]
	for j in range(0,len(line)):
		if(not((line[i],line[j]) in my_dic)):
			for key in my_dic:
				(a,b)=key
				

				











