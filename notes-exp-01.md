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
