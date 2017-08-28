import os, unicode, strutils, streams, tables, parsecsv, algorithm, random

proc main()=
    let line ="26s - loss: 0.3458 - acc: 0.8907 - val_loss: 0.3320 - val_acc: 0.896"

    let token="s - loss: "
    let k=line.find(token)
    let ll = line[(k+len(token)) .. ^1]
    let split_tokens = splitWhitespace(ll)
    for t in split_tokens:
        echo "[",t,"]"
    let tr_loss = parseFloat(split_tokens[0])
    let tr_acc  = parseFloat(split_tokens[3])
    let va_loss = parseFloat(split_tokens[6])
    let va_acc  = parseFloat(split_tokens[9])
main()
