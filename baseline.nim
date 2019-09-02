import os, unicode, strutils, streams, tables, parsecsv, algorithm 

const
    BASE_DIR = "./data"
    GLOVE_DIR = BASE_DIR & "/glove.6B/"
    EMBEDDINGS_FILE = "glove.6B.100d.txt"
    WANG_DATA_DIR = BASE_DIR & "/wang/"

# --------------------------------------------------------------
proc load_dataset(filename:string):(seq[string],seq[int]) =
    var p:CsvParser
    var 
        texts:seq[string]
        labels:seq[int]
    texts = @[]
    labels= @[]
    p.open(WANG_DATA_DIR & filename)
    p.readHeaderRow()
    while p.readRow():
        let qa = p.rowEntry("qtext") & 
                 " " & 
                 p.rowEntry("atext") 
        let li = parseInt(p.rowEntry("label"))
        texts.add(qa)
        labels.add(li)
#echo "text=\n",texts[0]
#echo "\nlabel=\n",labels[0]     
    p.close()
    return (texts,labels)
# --------------------------------------------------------------
proc main() =
    let (te,la) = load_dataset("train.csv")

    var 
        result0 = 0
        result1 = 0

    for ll in la:
        result0 += (if ll == 0: 1 else: 0)
        result1 += (if ll == 1: 1 else: 0)

    echo "answered all 1 yields accuracy =", result1/len(la)
    echo "answered all 0 yields accuracy =", result0/len(la)
# --------------------------------------------------------------
main()
