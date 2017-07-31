import os, unicode, strutils, streams, tables, parsecsv, algorithm, random

proc main()=
    let line ="Metrics-val: precision= 1.0 ,recall= 0.136363636364 ,f1= 0.24 ,avg_prec= 0.598404511713"

    let token="Metrics-val:"
    let k=line.find(token)
    let ll = line[(k+len(token)) .. ^1]
    let split_tokens = splitWhitespace(ll)
    for t in split_tokens:
        echo "[",t,"]"
    let precision = parseFloat(split_tokens[1])
    let recall    = parseFloat(split_tokens[3])
    let f1        = parseFloat(split_tokens[5])
    let avg_prec  = parseFloat(split_tokens[7])
main()



#[precision=]
#[1.0]
#[,recall=]
#[0.136363636364]
#[,f1=]
#[0.24]
#[,avg_prec=]
#[0.598404511713]
