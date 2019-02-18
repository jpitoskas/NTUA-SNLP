dict = {'any':'eh n iy', 'e':'iy', 'many':'m eh n iy', 'men':'m eh n', 'per':'p er', 'persons':'p er s uh n z'
,'lessons':'l eh s uh n z','sons':'s uh n z','suns':'s uh n z','sunset':'s uh n z eh t','to':'t uw'
,'tomb':'t uw m','too':'t uw','two':'t uw'}
filename = 'chars.syms'
filename = open(filename, 'w')
filename.write('<epsilon>'+ '\t\t' + str(0)+'\n')
symbol = 1
for key, value in dict.items():
    for i in value.split():
        line = i + "\t\t\t\t" + str(symbol) + '\n'
        filename.write(line)
        symbol+=1
    line = key + "\t\t\t\t" + str(symbol) + '\n'
    symbol+=1
    filename.write(line)
filename.close()

filename = 'chars.stxt'
filename = open(filename, 'w')
symbol = 2
for key, value in dict.items():
    line = value.split()
    filename.write('0 ' + str(symbol) + ' <epsilon> ' + line[0] +'\n')
    for i in range(1,len(line)):
        filename.write(str(symbol) + ' ' + str(symbol+1)+ ' <epsilon> '+line[i]+'\n')
        symbol +=1
    filename.write(str(symbol) + ' 1 ' + key + ' <epsilon> ' + '\n')
    symbol+=2

filename.write('1 0 <epsilon> <epsilon>'+'\n')
filename.write('1'+'\n')
filename.write(str(1))
filename.close()

# !make -s transducer

filename = 'phonem.txt'
filename = open(filename, 'w')
phonem = "s uh n z t uw m eh n iy p er".split()
state = 0
for letter in phonem:
    filename.write(str(state)+' '+str(state+1)+' '+letter+ '\n')
    state+=1

filename.write(str(state)+'\n')
filename.close()
# !make -s phonem


filename = 'newtransducer.stxt'
filename = open(filename, 'w')
symbol = 1
for key, value in dict.items():
    filename.write('0 ' + str(symbol) + ' '+ key + ' '+ key + ' 0\n')
    symbol+=1
transition_symbol = symbol
symbol = 1

for key, value in dict.items():
    for key1, value1 in dict.items():
        if(key == key1):
            filename.write(str(symbol) +' '+ str(symbol) + ' '+ key + ' ' + key +' 0\n')
        else:
            filename.write(str(symbol)+ ' ' + str(transition_symbol)+ ' ' + key1 + ' '+ key1 + ' ' + str(abs(len(key) - len(key1)))+' \n')
            transition_symbol+=1
        state_return = transition_symbol % 14
        filename.write(str(transition_symbol) + ' ' + str(state_return) + ' ' + '<epsilon> <epsilon> 0 \n')
    symbol+=1

for i in range(1,14):
    filename.write(str(i)+'\n')
filename.close()
# !make -s newtransducer
