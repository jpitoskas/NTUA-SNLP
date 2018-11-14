def format_arc(src, dst, src_sym, dst_sym, w):
    # out = open('test.fst', 'w')
    # out.write(str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')
    # out.close()
    return (str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')


# from lib import *

acceptor = []
s = 1

letters = list('projact')

for i in range(0, len(letters)):
    print(
        format_arc(
            src=s, dst=s+1, src_sym=letters[i], dst_sym=letters[i], w=0))
    s += 1
    if i == len(letters) - 1:
        print(
            format_arc(
                src=s, dst=0, src_sym='<epsilon>', dst_sym='<epsilon>', w=0))
print(0)
