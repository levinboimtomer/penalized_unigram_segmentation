from collections import defaultdict

__author__ = 'tomerlevinboim'
import SegUtil
import MonotoneFSTUtil
import numpy as np


class Record(object):
    pass


def readlines(filename, rstrip=False):
    lines = []
    with open(filename, 'rb') as f:
        for (i, line) in enumerate(f):
            if rstrip:
                line = line.rstrip()
            lines.append(line)

    return lines

############################################


def Model(Pseg=None):
    R = Record()
    R.LL = -np.inf
    R.Pseg = defaultdict(lambda: 0) if Pseg is None else Pseg
    return R


def init_model(data, params=None):
    if params is None:
        return Model()
    else:
        exp_counts = Model(SegUtil.count_segments(data, params))
        # globals.most_common(exp_counts.Pseg, k=100)
        return M_step(exp_counts)


def M_step(ec):
    # compute the next model from the expected counts
    # by applying count-and-divide
    total = sum([ec.Pseg[seg] for seg in ec.Pseg])

    if total <= 0 or np.isnan(total):
        print '## Warning: total is', total

    Pseg = {seg: (ec.Pseg[seg]/total) for seg in ec.Pseg}
    return Model(Pseg)


def E_step(model, data):
    # initialize the expected counts
    Pseg = defaultdict(lambda: 0)
    LL = 0

    for i, line in enumerate(data.lines):       # go over each sentence and collect expected counts
        logprob_i, counts_i = MonotoneFSTUtil.forward_backward(model, params, line)
        LL += logprob_i

        for seg in counts_i:
            Pseg[seg] += counts_i[seg]      # accumulate the expected counts
        #print '  abc = ', Pseg['abc']
        #print '  lola = ', Pseg['lola']

    ec = Model(Pseg)
    return LL, ec


def viterbi_decode(data, params, model):
    decodes = []
    for i, line in enumerate(data.lines):
        decode_i = MonotoneFSTUtil.viterbi(model, params, line, line_no=i)
        I = decode_i[0]
        decodes.append(SegUtil.segment_str(line, I)+ list(decode_i))
    return decodes


def EM_train(data, params, model):
    for iter in xrange(params.MAX_ITER):
        # E-step
        LL, exp_counts = E_step(model, data)
        print "E-step", iter, " LL =", LL

        # M-step
        model = M_step(exp_counts)
        print "M-step", iter

        if model.LL > LL:
            print "Warning: log likelihood decreased!"
        if np.isclose(model.LL, LL, 1e-10):
            print "Converged at iteration", iter
            break
        model.LL = LL

    return model


def load_data(filename, remove_whitespace=False, chop=-1):
    data = Record()
    if filename == 'test':
        data.lines = ['abc d', 'run lola']
        # uncomment the line "P = normalize..." in the forward_backward() function
        # set beta = 1,
        # then
        # running with ['abc d'] leads to the following alpha/beta values
        # alpha = array([1.0000, 0.3333, 0.5000, 1.0000, 1.0000, 1.0000])
        # beta = array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]) (this makes sense because normalization makes the lattice stochastic)
    else:
        original_lines = readlines(filename, rstrip=True)
        if remove_whitespace:
            original_lines = [line.replace(' ', '') for line in original_lines]
        if chop > 0:
            original_lines = [line[:chop] for line in original_lines]

        data.lines = original_lines

    return data


def init_params(MAX_ITER=4, MAX_SEG_LENGTH=5, beta=1.6):
    params = Record()
    params.MAX_ITER = MAX_ITER
    params.MAX_SEG_LENGTH = MAX_SEG_LENGTH
    # set beta = 1 for debug
    params.beta = beta  # Liang et al. (2006): For English, we used beta = 1.6; Chinese, beta = 2.5.
    params.betas = [1.0] + [ ((i*1.0) ** -params.beta) for i in xrange(1, params.MAX_SEG_LENGTH + 1) ]
    return params


if __name__ == '__main__':
    filename = 'data/es-en.50.en'
    #filename = 'test'

    params = init_params(MAX_ITER=10, MAX_SEG_LENGTH=5, beta=0)
    data = load_data(filename, remove_whitespace=True, chop=20)

    model0 = init_model(data, params)
    model = EM_train(data, params, model0)
    decodes = viterbi_decode(data, params, model)
