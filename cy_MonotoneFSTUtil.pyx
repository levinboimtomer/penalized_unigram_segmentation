#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
import numpy as np
cimport numpy as np
from SegUtil import segments
from libc.math cimport exp, log

def newTransitionMatrix(object model, object params, int N, str line):
    cdef size_t i, j, N_seg
    cdef str seg
    cdef float p, b_N
    cdef np.ndarray[double, ndim=1] betas
    cdef dict Pseg

    Pseg = model.Pseg

    betas = params.betas
    P = np.zeros([N, N])
    for i in xrange(N - 1):
        P[i, i + 1] = 1     # assume P[i,i+1] = 1 (required so that we skip over ' ' with probability 1)
    for (i, j, seg) in segments(line, params.MAX_SEG_LENGTH):
        N_seg = len(seg)
        b_N = betas[N_seg]
        p = Pseg[seg]
        P[i, j] = p * b_N

    return P

cdef float NEG_INF = -np.inf
cdef float logsumexp(np.ndarray[double, ndim=1] a, size_t minI, size_t maxI):
    cdef size_t i
    cdef float result = 0.0
    cdef float M = NEG_INF

    for i in xrange(minI, maxI+1):
        if a[i] > M:
            M = a[i]
    if M == NEG_INF:
        return NEG_INF

    for i in xrange(minI, maxI+1):
        result += exp(a[i] - M)
    return M + log(result)


cpdef cy_forward(object params, int N, object logP):
    cdef np.ndarray[double, ndim=1] s
    cdef np.ndarray[double, ndim=1] log_alpha
    cdef int j
    cdef size_t n, K
    cdef float v

    log_alpha = NEG_INF * np.ones(N)
    log_alpha[0] = 0  # log(1)
    K = params.MAX_SEG_LENGTH
    for n in xrange(1, N):
        s = log_alpha + logP[:, n]
        j = n-K-1
        if j < 0: j = 0
        v = logsumexp(s, minI=j, maxI=n)
        log_alpha[n] = v

        #v2 = sm.logsumexp(s) # takes much more time than the above implementation
        #assert np.isclose(v, v2), (v, v2, s, log_alpha, logP[:, n])

    return log_alpha


def cy_backward(object params, int N, object logP):
    alpha = cy_forward(params, N, np.fliplr(np.flipud(logP.T)))           # transpose and then left-right and up-down flip P
    beta = alpha[::-1]                                          # reverse
    return beta


def collect_counts(line, object params, object logP, double[:] log_alpha, double[:] log_beta, double Z):
    cdef size_t i, j
    cdef dict C
    cdef float edge_posterior, c_seg, a_i, b_j, e_ij

    C = dict()  # C[seg] stores expected counts for segment seg
    for (i, j, seg) in segments(line, params.MAX_SEG_LENGTH):
        a_i = log_alpha[i]
        e_ij = logP[i, j]
        b_j = log_beta[j]
        edge_posterior = exp(a_i + e_ij + b_j - Z)  # alpha[i]*P_ij*beta[j]/Z
        # assert edge_posterior >= 0 and edge_posterior <= 1
        if seg in C:
            C[seg] += edge_posterior
        else:
            C[seg] = edge_posterior
    return C


# _viterbi_forward() is similar to the _forward() function,
# however, instead of computing the total probability of getting to state i from all previous states
# we only maintain the probability of the best reaching path (to a state) as well as index pointers for backtracking
cpdef cy_viterbi_forward(size_t N, object logP):
    cdef np.ndarray[int, ndim=1] back
    cdef np.ndarray[double, ndim=1] log_v, logPn
    cdef list u
    cdef size_t n, k, best_j
    cdef float v

    back = np.zeros(N, dtype=np.int32)                          # backtrack indices
    log_v = NEG_INF*np.ones(N)                                  # score of best path to state.
    log_v[0] = 0                                                # = log(1)

    for n in xrange(1, N):
        logPn = logP[:, n]
        u = [log_v[k] + logPn[k] for k in xrange(n+1)]          # TODO: might want to take into account params.MAX_SEG_LENGTH
        best_j = np.argmax(u)                                   # index of preceding state
        back[n] = best_j
        log_v[n] = u[best_j]                                    # value of best path to n

    return log_v, back