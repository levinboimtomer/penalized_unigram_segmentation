__author__ = 'tomerlevinboim'
#from nltk.util import ngrams
from collections import defaultdict


def count_segments(data, params):
    # segments all the N-grams up to length N<=params.MAX_SEG_LENGTH
    Pseg = defaultdict(lambda: 0)
    for line in data.lines:
        # for n in xrange(1, params.MAX_SEG_LENGTH+1):
        #     for ngram in ngrams(line, n=n): # change to segments()
        #         ngram_str = ''.join(ngram)
        #         k = len(ngram_str)
        #         segments[ngram_str] += params.betas[k]

        for i, j, seg in segments(line, params.MAX_SEG_LENGTH):
            k = len(seg)
            Pseg[seg] += 1.0  # params.betas[k]

    return Pseg


# return all ngrams in line of length up to K
def segments(line, K):
    N = len(line)+1
    for i in xrange(N):
        M = min(N, i+1+K)
        for j in xrange(i+1, M):
            seg = line[i:j]
            if ' ' in seg:
                break
            yield i, j, line[i:j]


# segments s according to the index in I
# s='asbcd'; I=[0,2,5]
# outputs: ['as', 'bcd']
def segment_str(s, I):
    R = xrange(len(I)-1)
    return [ s[I[k]:I[k+1]] for k in R]