# -*- coding: utf-8 -*-

def parse_output(filename):
    with open(filename,'r') as outfile:
        lines = outfile.readlines()
    
    content = [line.strip() for line in lines if 'val_loss' in line]

    #"3775/3775 [==============================] - 8s - loss: 0.2554 - acc: 0.9290 - val_loss: 0.2678 - val_acc: 0.9152"
    tr_loss = []
    tr_accu = []
    va_loss = []
    va_accu = []
    token='[==============================]'
    for line in content:
        k = line.find(token)
        ll = line[k+len(token):]
        tokens = ll.split()
        # print tokens[4],tokens[7],tokens[10],tokens[13]
        tr_loss.append(float(tokens[4]))
        tr_accu.append(float(tokens[7]))
        va_loss.append(float(tokens[10]))
        va_accu.append(float(tokens[13]))

    return(tr_loss,tr_accu,va_loss,va_accu)

t_l,t_a,v_l,v_a = parse_output('output.txt')
print t_l,t_a,v_l,v_a
    
