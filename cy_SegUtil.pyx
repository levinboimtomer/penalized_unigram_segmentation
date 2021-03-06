#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
import numpy as np
cimport numpy as np

# return all ngrams of length up to K within line.
# Note: an ngram cannot contain space.
def segments(str line, size_t K):
    cdef size_t i, j, N, M
    cdef str seg

    N = len(line)+1
    for i in xrange(N):
        M = min(N, i+1+K)
        for j in xrange(i+1, M):
            seg = line[i:j]
            if ' ' in seg:
                break
            yield i, j, seg