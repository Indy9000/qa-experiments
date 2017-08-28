import os, times, unicode, strutils, streams, tables, parsecsv, algorithm, random, future

type
  TStats = tuple[ttp:int, tfp:int, ttn:int, tfn:int,
                 vtp:int, vfp:int, vtn:int, vfn:int,
                 filter_count:int, kernel_size:int, pooling_windows:int,
                 dense_size1:int, dropout1:float, dense_size2:int, dropout2:float,
                 batch_count:int,
                 overfitted:bool, hyper_params:string,
                 test_mrr:float, val_mrr:float,
                 test_map:float, val_map:float, filename:string]

proc parse_stats(filename:string):TStats =
  let token0="- val_loss: "
  let token1="TEST tp="
  let token2="VALIDATION tp="
  let token5="filter-count =  "

  var val_loss1 = -1.0
  var val_loss2 = -1.0
  var val_acc1 = -1.0
  var val_acc2 = 1000.0

  var stats:TStats
  
  for line in lines filename:
    ######## Overfit check ########
    let k0 = line.find(token0)
    if k0 > -1:
      let ll = line[(k0+len(token0)) .. ^1]
      let split_tokens = splitWhitespace(ll)
      #echo "SPLIT TOKENS=", $(split_tokens)
      
      if val_loss1 == -1.0:
        val_loss1=parseFloat(split_tokens[0])
      else:
        val_loss2=parseFloat(split_tokens[0])

      if val_acc1 == -1.0:
        val_acc1 = parseFloat(split_tokens[3])
      else:
        val_acc2 = parseFloat(split_tokens[3])

    ###############################

    let k1 = line.find(token1)
    if k1 > -1:
      #echo "token1 found"
      let ll = line[(k1+len(token1)) .. ^1]
      let split_tokens = splitWhitespace(ll)
      #echo "split_token len",len(split_tokens), split_tokens

      stats.ttp = parseInt(split_tokens[0])
      stats.tfp = parseInt(split_tokens[2])
      stats.ttn = parseInt(split_tokens[4])
      stats.tfn = parseInt(split_tokens[6])
      stats.filename = filename
      
    let k2 = line.find(token2)
    if k2 > -1:
      #echo "token2 found"
      let ll = line[(k2+len(token2)) .. ^1]
      let split_tokens = splitWhitespace(ll)
      stats.vtp = parseInt(split_tokens[0])
      stats.vfp = parseInt(split_tokens[2])
      stats.vtn = parseInt(split_tokens[4])
      stats.vfn = parseInt(split_tokens[6])
      stats.filename = filename
      #echo valStats.filename, testStats.filter2.nim
    
    let k5 = line.find(token5)
    if k5 > -1:
      stats.hyper_params = line
      let ll=line[k5+len(token5) .. ^1]
      let split_token = splitWhitespace(ll)
      try:
        stats.filter_count    = parseInt(split_token[0])
        stats.kernel_size     = parseInt(split_token[2])
        stats.pooling_windows = parseInt(split_token[4])
        if split_token[5] == "dense-size1":
          stats.dense_size1     = parseInt(split_token[6])
          stats.dense_size2     = parseInt(split_token[8])
          stats.dropout1        = parseFloat(split_token[10])
          stats.dropout2        = parseFloat(split_token[12])
          stats.batch_count     = parseInt(split_token[14])
        else:
          stats.dense_size1     = parseInt(split_token[6])
          stats.dropout1        = parseFloat(split_token[8])
          stats.batch_count     = parseInt(split_token[10])
      except:
        echo "Error: ",$(split_token)

  if val_loss2 > val_loss1 or val_acc2 < val_acc1:
    stats.overfitted = true
  
  return stats

proc parse_line(line:string):(float,float) =
    let token3="CorrectAnswers"
    let token4="PredictedList"

    let k3 = line.find(token3)
    if k3 > -1:
        var items = line.split({ '(', ',', ')', '[', ']' })
        #echo $(items)
        var ca_found = false
        var pr_found = false
        
        var correct_indices:seq[int]
        correct_indices = @[]
        
        var predicted_s: seq[string]
        predicted_s = @[]
        
        for t in items:
            if t.strip() == "" :
                continue
            if token3 in t:
                ca_found = true; continue
            if token4 in t:
                pr_found = true; continue
            if ca_found and not pr_found:
                correct_indices.add(parseInt(t.strip()))
            if pr_found:
                predicted_s.add(t.strip())
        
        #echo "Predicted_S=",predicted_s

        type
            Item = tuple[index:int, value:float] 
        var predicted:seq[Item]
        predicted = @[]
        for i in 0..int(len(predicted_s)/2)-1:
            #extract tuples
            #echo ">>>",predicted_s[2*i],">>>",predicted_s[2*i+1]
            var ii:Item
            ii.index = parseInt(predicted_s[2*i].strip())
            ii.value = parseFloat(predicted_s[2*i+1].strip())
            predicted.add(ii)
        
        #echo "correctA=",correct_indices,"\n\n"
        #echo "Predicted=",predicted
        predicted.sort((a,b:Item)=> cmp(a.value,b.value), order=Descending)
        #echo "Predicted=",predicted

        #compute MRR
        var rank = 1
        var rr = 0.0
        for it in predicted:
            if it.index in correct_indices:
                rr += 1.0/float(rank)
                break
            rank += 1

        #compute MAP
        rank = 1
        var hits = 0
        var ap = 0.0
        for it in predicted:
            if it.index in correct_indices:
                hits += 1
                ap += float(hits)/float(rank)
            rank += 1
        if len(correct_indices)>0:
            ap /= float(len(correct_indices))
        
        return (rr,ap)



type
  QPerf = tuple[qid:string,rr:float,ap:float,qclass:string]


# Parses the results and computes the MAP and MRR
proc parse_results(filename:string):seq[QPerf] =
  let token1 ="for validation set"
  let token2 ="for test set"
  let token3 = "QID"

  var val_found =false
  var test_found =false
  
  var val_map = (float)0.0
  var val_mrr = (float)0.0
  var val_count = (int)0

  var test_map = (float)0.0
  var test_mrr = (float)0.0
  var test_count = (int)0


  var qperf_list:seq[QPerf]
  qperf_list = @[]

  for line in lines filename:
    if token1 in line:
      val_found = true
      continue
    if token2 in line:
      test_found = true
      continue
  
    if token3 in line:
      if val_found and not test_found:
        let (rr,ap) = parse_line(line)
        val_mrr += rr
        val_map += ap
        val_count += 1

      if test_found:
        let (rr,ap) = parse_line(line)
        test_mrr += rr
        test_map += ap
        test_count += 1

        let split_tokens = splitWhitespace(line)
        var qp:QPerf
        qp.qid= split_tokens[1]
        qp.rr = rr
        qp.ap = ap
        #echo "QP ", qp.qid," ", qp.rr," ",qp.ap
        qperf_list.add(qp)


  return qperf_list

proc main() =
  if paramCount() != 1:
    quit("synopsis: " & getAppFilename() & " [exp-results-file-to-parse]")

  var qclass_list:seq[string]
  qclass_list = @[]

  for line in lines "who-what-when-test.txt":
    qclass_list.add(line)

  let filename = paramStr(1)
  echo "loading: " & filename
  var i = 0
  var qperf_list = parse_results(filename)

  for i in 0..95:
    qperf_list[i].qclass = qclass_list[i]

  #sort qperf descending according to ap value
  qperf_list.sort((a,b:QPerf)=> cmp(a.ap, b.ap), order=Descending)
  for qp in qperf_list:
    echo qp.ap," ",qp.qclass

  if(false):
    echo "------------"
    let ss = len(qperf_list)
    for i in ss-16 .. ss-7:
      echo $(qperf_list[i])

    echo "============"
    for i in int(ss/2)-5 .. int(ss/2)+5:
      echo $(qperf_list[i])

#-------------------------------------------------------
main()

