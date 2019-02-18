import sys
orig_stdout=sys.stdout
out=open('tr1.txt','w')
sys.stdout=out
a=[]
a=['T','E','T','C','G','A','G']
h=0

for i in range(0,len(a)):
	print(h,h+1,a[i],a[i],0)
	h=h+1
print(h)

sys.stdout=orig_stdout
out.close()
