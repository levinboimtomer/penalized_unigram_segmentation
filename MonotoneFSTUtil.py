import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cy_MonotoneFSTUtil import *

# Computing forward backward and Viterbi for monotone FSTs
# which are FSTs whose state transition matrix is upper trianguar
# compute the forward-backward and vweight along the lattice


# makes an upper triangular matrix P[i, j]=probability of moving from state i to state j.

def forward_backward(model, params, line):
    N = len(line) + 1
    logP = np.log(newTransitionMatrix(model, params, N, line), dtype=np.float32)  # compute the (log-)transition matrix

    log_alpha = cy_forward(params, N, logP)                               # compute forward weights
    log_beta = cy_backward(params, N, logP)                               # compute backward weights
    assert np.isclose(log_alpha[-1], log_beta[0]), "alpha=%f, beta=%f" % (log_alpha[-1], log_beta[0])
    Z = log_alpha[-1]                                                     # Z = prob(line)

    C = collect_counts(line, params, logP, log_alpha, log_beta, Z)

    return Z, C

# _viterbi_forward() is similar to the _forward() function,
# however, instead of computing the total probability of getting to state i from all previous states
# we only maintain the probability of the best reaching path as well as index pointers for backtracking
# private
def _viterbi_forward(N, logP):
    back = np.zeros(N, dtype=int)                               # backtrack indices
    log_v = -np.inf*np.ones(N)                                  # score of best path to state.
    log_v[0] = 0                                                # = log(1)

    for n in xrange(1, N):
        logPn = logP[:, n]
        u = [log_v[k] + logPn[k] for k in xrange(n+1)]  # TODO: might want to take into account params.MAX_SEG_LENGTH
        best_j = np.argmax(u)                                   # index of preceding state
        back[n] = best_j
        log_v[n] = u[best_j]                                    # value of best path to n

    return log_v, back


# computes the Viterbi segmentation of a given line
def viterbi(model, params, line, line_no=None):
    N = len(line) + 1
    logP = np.log(newTransitionMatrix(model, params, N, line))  # transpose and then left-right and up-down flip P

    log_v, back = _viterbi_forward(N, logP)                     # compute viterbi path and score

    I = []                                                      # backtrack through back[]
    j = N - 1
    while j != 0:
        I.append(j)
        j = back[j]  # backtrack
        assert j >= 0

    J = [0] + I[::-1]

    return J, log_v[-1]



## some interactive debugging code
# P = normalize(P, norm='l1', axis=1, copy=False)  # TL: uncomment for debugging
# np.set_printoptions(formatter={'float': lambda x: '   -  ' if x == 0 else '%2.4f' % x})
# print np.matrix(P)


## the first _forward() function, before working in log-probability
# def _forward():
#     alpha = np.zeros(N, )
#     alpha[0] = 1
#     for n in xrange(1, N):
#         alpha[n] = alpha.dot(P[:, n])
#         #print alpha
#     return alpha