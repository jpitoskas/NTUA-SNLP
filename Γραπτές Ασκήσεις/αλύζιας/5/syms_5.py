import sys
orig_stdout=sys.stdout
out=open('I.syms','w')
sys.stdout=out
print("<eps>",0)
print("A",1)
print("G",2)
print("C",3)
print("T",4)
print("E",5)
print("F",6)

sys.stdout=orig_stdout
out.close()

