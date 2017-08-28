import os, times, unicode, strutils, streams, tables, parsecsv, algorithm, random, future

type
  TStats = tuple[ttp:int, tfp:int, ttn:int, tfn:int,
                 vtp:int, vfp:int, vtn:int, vfn:int,
                 filter_count:int, kernel_size:int, pooling_windows:int,
                 dense_size1:int, dropout1:float, dense_size2:int, dropout2:float,
                 batch_count:int,
                 overfitted:bool, hyper_params:string,
                 test_map:float, val_map:float, filename:string]

proc parse_stats(filename:string):TStats =
  let token0="- val_loss: "
  let token1="TEST tp="
  let token2="VALIDATION tp="
  let token3="Test Mean avg Prec = "
  let token4="Val Mean avg Prec = "
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
    
    let k3 = line.find(token3)
    if k3 > -1:
      let ll=line[k3+len(token3) .. ^1]
      let split_token = splitWhitespace(ll)
      stats.test_map = parseFloat(split_token[0])

    let k4 = line.find(token4)
    if k4 > -1:
      let ll=line[k4+len(token4) .. ^1]
      let split_token = splitWhitespace(ll)
      stats.val_map = parseFloat(split_token[0])
    
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

proc main() =
  if paramCount() != 2:
    quit("synopsis: " & getAppFilename() & " [exp-results-folder-to-parse] [results_count]")

  let folder = paramStr(1)
  let results_count= parseInt(paramStr(2))
  
  echo "enumerating folder: " & folder
  var
    stats:seq[TStats]
  stats = @[]
    
  for f in walkFiles(folder&"/*.txt"):
    let st = parse_stats(f)
    if not st.overfitted:
      stats.add(st)


  stats.sort((a,b:TStats)=> cmp(a.val_map, b.val_map), order=Descending)
  echo "sorted by val_map"
  var count=0
  for st in stats:
    if st.ttp > 0 and st.tfp > 0 and st.vtp > 0 and st.vfp > 0:
      #echo $(st)
      echo st.filter_count,"&", st.kernel_size, "&",st.pooling_windows,"&", st.dense_size1,"&", st.dropout1,"&", st.dense_size2,"&", st.dropout2,"&", st.batch_count, "&",st.test_map,"&", st.val_map, "\t", st.filename
      count += 1
    if count >= results_count:
      break;

  if(false):
    echo "\nsorted by test_map"
    stats.sort((a,b:TStats)=> cmp(a.test_map, b.test_map), order=Descending)
    count=0
    for st in stats:
      if st.ttp > 0 and st.tfp > 0 and st.vtp > 0 and st.vfp > 0:
        echo $(st)
        count += 1
      if count >= results_count:
        break;

main()

