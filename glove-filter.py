from __future__ import print_function

import os
import sys
import numpy as np
import csv


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)

def cleanup_embedding(src_embedding_file, dst_embedding_file):
    print('Loading word vectors.')
    word_counter = 0
    removed_counter = 0
    with open(dst_embedding_file,'w') as fo:
        with open(src_embedding_file,'r') as fi:
            for line in fi:
                values = line.split()
                word = values[0]
                if has_numbers(word):
                    print("removing ", word)
                    removed_counter += 1
                else:
                    word_counter += 1
                    fo.write(line) 
    print('Finished, words added:', word_counter, " removed:", removed_counter)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("This cleans up embeddings by filtering words that contains numbers\n", \
              "Command line: ", sys.argv, " src-embedding-file dst-embedding-file", \
             ) 
        exit()
    else:
        cleanup_embedding(sys.argv[1],sys.argv[2])


