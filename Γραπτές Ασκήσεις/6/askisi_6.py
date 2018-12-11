import sys
from collections import defaultdict
orig_stdout=sys.stdout
out = open('chars.syms','w')
input_file= open('filr.txt','r')
line=input_file.readlines()
sys.stdout = out
my_dict={}
print('<eps>',0)
h=1;

for i in range(0,len(line)):
	word=line[i].split()
	for i in range(0,len(word)):
		my_dict[word[i]]=word[i]
for key in my_dict:
	print(key,h)
	h=h+1



sys.stdout = orig_stdout
out.close()
input_file.close()

orig_stdout=sys.stdout
input_file= open('filr.txt','r')
out = open('transducer.txt','w')
lines = input_file.readlines()
final=[]
sys.stdout=out
teliko=1;
for i in range(0,len(lines)):

	start=0;
	flag=0;
	word = lines[i].split()
	for i in range(0,len(word)):

		if (i==1):
			print(start,teliko,word[i],word[0]);
			start=teliko
			teliko=teliko+1
		if(i>1):
			print(start,teliko,word[i],'<eps>')
			start=teliko
			teliko=teliko+1
	final.append(teliko-1)
for i  in range(0,len(final)):
	print(final[i])


sys.stdout = orig_stdout
input_file.close()
out.close()
