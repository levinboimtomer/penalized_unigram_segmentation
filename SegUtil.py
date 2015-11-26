import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cy_SegUtil import *


def count_segments(data, params):
    # segments all the N-grams up to length N<=params.MAX_SEG_LENGTH
    Pseg = dict()
    for line in data.lines:
        # for n in xrange(1, params.MAX_SEG_LENGTH+1):
        #     for ngram in ngrams(line, n=n): # change to segments()
        #         ngram_str = ''.join(ngram)
        #         k = len(ngram_str)
        #         segments[ngram_str] += params.betas[k]

        for i, j, seg in segments(line, params.MAX_SEG_LENGTH):
            k = len(seg)
            b_k = params.betas[k]
            Pseg[seg] = b_k if seg not in Pseg else b_k + Pseg[seg]

    return Pseg





# segments s according to the index in I
# s='asbcd'; I=[0,2,5]
# outputs: ['as', 'bcd']
def segment_str(s, I):
    R = xrange(len(I)-1)
    return [ s[I[k]:I[k+1]] for k in R]