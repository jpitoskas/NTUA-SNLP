import sys
orig_stdout=sys.stdout
out=open('I.txt','w')
sys.stdout=out
a=[]
a=['A','G','C','T','E','F']
h=0
for i in range(0,len(a)):
	print(0,h+1,a[i],a[i],0)
	h=h+1
for i in range(0,len(a)):	
	print(0,h+1,a[i],'<eps>',1)
	h=h+1
for i in range(0,len(a)):	
	print(0,h+1,'<eps>',a[i],1)
	h=h+1
for i in range(0,len(a)):	
	for j in range(0,len(a)):
		if(a[i]!=a[j]):	
			print(0,h+1,a[i],a[j],1)
			h=h+1
for i in range(1,h):
	print(i)

sys.stdout=orig_stdout
out.close()

