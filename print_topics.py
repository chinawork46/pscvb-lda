import sys
import numpy as np

corpus_file = sys.argv[1]
phi_file = sys.argv[2]

f = open(corpus_file)
s = f.read()
f.close()

corpus = [''] + [i.strip() for i in s.split() if i.strip() != '']

f = open(phi_file)
lines = f.readlines()
f.close()

k = len([i for i in lines[0].split() if i.strip() != ''])
w = len(lines)
phi = np.zeros((w, k))

for i in xrange(w):
     line = lines[i]
     its = [float(j) for j in line.split() if j.strip() != '']
     phi[i] = np.array(its)

print np.sum(phi)

for i in xrange(k):
     words = list(phi[:, i])
     idx_words = []
     
     for j in xrange(w):
          idx_words.append((j, words[j]))
 
     idx_words = sorted(idx_words, key=lambda k: k[1], reverse=True)
     
     print i, ':', 
     norm = np.sum(words)
     
     for idx, value in idx_words[:10]:
          print corpus[idx + 1], #value / norm, 

     print '\n'
