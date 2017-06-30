Experiment notes 
================

2017-06-19

Embeddings glove.6b.zip 100d doesn’t seem to contain most of the words in the dataset
Namely 11145/11615 is missing. 

Processing text dataset
Found 4718 samples
max seq len= 66
Found 11615 unique tokens
Shape of data tensor: (4718, 70)
Shape of label tensor: (4718, 2)
Shape of training samples (3775, 70)
Shape of training labels (3775, 2)
Shape of validation samples (943, 70)
Shape of validation labels (943, 2)
Loading word vectors.
Found 400000 word vectors.
words without embeddings = 11145
len(labels) 4718
Train on 3775 samples, validate on 943 samples

Sample of words not present in the embeddings:
privileges, buddhist, radiation, cross, member, adverse, creil, largest, units, gets, privileged, difficult, spirited, premiere, prasong, diplomat, coaches, student, poznan, koryo, lobby, correspondent, barum, whole, purposefully, fighting, premiership, theses, shimbun, english, firstly, conquer, breaker, sensing, realised, upgraded, routinely, looked, rocket, heavily, khrushchev, aliso, obtain, mfn, cries, fishing, tci, happiness, disturbance, console, serbs, supply, sky, capabilities, hall, book, adoption, attractive, earnestly, smart, ski, unprecedented, enact, identical, washing, saturation, tizzy, sick, provisions, execute, interrogation, know, rajasthan, press, racists, sk1, loses, overtures, washingtonian, hosts, january, adjoining, repressed, branca, wonders, alabama, exceed, because, smoothly, dianetics, 115, 117, growth, export, openness, ablaze, empire, rossini, lead, receivership, breakthroughs, leaded, mines, leap, mined, leader, ripped

-------------------------------------------------------------------------------
2017-06-22

Embeddings contain a large number of numbers and dates and other numerical values. These are very unlikely to be useful but takes up memory when loaded on to a matrix. In glove.42b.300d embeddings we removed 529649 such tokens. This reduced the token count to 1387845. Following is an example of tokens removed:

xb7, xdarklegacyx2, xj1100, xj75, xl75, xml-commons-external-1, xp80, xs/1000d, xs67, xxxxx10x.xxx, y04, y510a, year25, yes225, yes235, yp-u3, z3/4, z3c, z5600, z6pa, z890, z999, zat-970a, zb4i

Further reduction of the token count is necessary for it to successfully load into a numpy matrix

-------------------------------------------------------------------------------
2017-06-27
It turns out that the missing words in the embeddings were due to a bug in the code. over 95% of the vocabulary is contained in the glove.6b.100d embeddings.



