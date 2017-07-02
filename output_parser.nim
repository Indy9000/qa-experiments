import os, unicode, strutils, streams, tables, parsecsv, algorithm

proc parse_output(filename:string) : tuple[tl:seq[float],ta:seq[float],
                                          vl:seq[float],va:seq[float]] = 
  var tr_loss = @[1.0]
  var tr_acc  = @[0.0]
  var va_loss = @[1.0]
  var va_acc  = @[0.0]

  for line in lines filename:
    if "val_loss" in line:
      let li = line.strip()
      let token="[==============================]"
      let k = li.find(token)
      if k > -1:
        let ll = li[(k+len(token)) .. ^1]
        let tokens = splitWhitespace(ll)
        echo "tr_loss ",tokens[4]," tr_acc ",tokens[7]," va_loss ",tokens[10]," va_acc ",tokens[13]
        tr_loss.add(parseFloat(tokens[4]))
        tr_acc.add(parseFloat(tokens[7]))
        va_loss.add(parseFloat(tokens[10]))
        va_acc.add(parseFloat(tokens[13]))

  return (tr_loss,tr_acc,va_loss,va_acc)

proc evaluate_performance(tl:seq[float], ta:seq[float],
                          vl:seq[float], va:seq[float]):float =
  for vv in 0..len(ta)-1:
    if ta[vv] >= 0.98:
      let tl = if tl[vv] > 1.0: 1.0 else: (1.0-tl[vv])
      let vl = if vl[vv] > 1.0: 1.0 else: (1.0-vl[vv])
      return ta[vv] * tl * va[vv] * vl
  let ee = len(ta)-1
  return ta[ee] * (1.0 - tl[ee]) * va[ee] * (1.0 - vl[ee])

proc main() =
  if paramCount() != 1:
    quit("synopsis: " & getAppFilename() & " exp-output-filename")
  let filename = paramStr(1)
  let (tl,ta,vl,va) = parse_output(filename)
  
  let fitness = evaluate_performance(tl,ta,vl,va)
  echo "fitness ", fitness

main()


# def parse_output(filename):
#   with open(filename,'r') as outfile:
#     lines = outfile.readlines()
#
#     content = [line.strip() for line in lines if 'val_loss' in line]
#
#     #"3775/3775 [==============================] - 8s - loss: 0.2554 - acc: 0.9290 - val_loss: 0.2678 - val_acc: 0.9152"
#     tr_loss = []
#     tr_accu = []
#     va_loss = []
#     va_accu = []
#     token='[==============================]'
#     for line in content:
#       k = line.find(token)
#       ll = line[k+len(token):]
#       tokens = ll.split()
#       # print tokens[4],tokens[7],tokens[10],tokens[13]
#       tr_loss.append(float(tokens[4]))
#       tr_accu.append(float(tokens[7]))
#       va_loss.append(float(tokens[10]))
#       va_accu.append(float(tokens[13]))
#
#   return(tr_loss,tr_accu,va_loss,va_accu)
#
#t_l,t_a,v_l,v_a = parse_output('output.txt')
#                                                                                                                 print t_l,t_a,v_l,v_a
