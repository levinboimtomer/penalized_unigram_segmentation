__author__ = 'tomerlevinboim'
from collections import defaultdict

import sys
import numpy as np

import SegUtil
from Util import *
import MonotoneFSTUtil


class Record(object):
    pass


def Model(Pseg=None):
    R = Record()
    R.LL = -np.inf
    R.Pseg = defaultdict(lambda: 0) if Pseg is None else Pseg
    return R


def init_params(MAX_ITER=4, MAX_SEG_LENGTH=5, beta=1.6, beta_offset=0):
    params = Record()
    params.MAX_ITER = MAX_ITER
    params.MAX_SEG_LENGTH = MAX_SEG_LENGTH
    # set beta = 1 for debug
    params.beta = beta  # Liang et al. (2006): For English, we used beta = 1.6; Chinese, beta = 2.5.
    #params.betas = [1.0] + [ -((z*1.0) ** params.beta) for z in xrange(1, params.MAX_SEG_LENGTH + 1) ]

    R = np.abs(beta_offset-np.array(range(0, params.MAX_SEG_LENGTH + 1)))
    params.betas = np.exp(-np.power(R, beta))

    return params


class Segmenter:

    def __init__(self, opts):
        self.params = init_params(MAX_ITER=opts.ITERATIONS, MAX_SEG_LENGTH=opts.MAX_SEG_LENGTH, beta=opts.beta, beta_offset=opts.offset)
        self.data = Record()
        self.filename = Record()


    def loadData(self, filename_train, filename_dev=None, remove_whitespace=False, chop=-1):
        self.filename.train = filename_train
        self.filename.dev = filename_dev
        self.params.remove_whitespace = remove_whitespace
        self.params.chop = chop

        if filename_train == 'test':
            self.data.lines = ['abc d', 'run lola']
            # uncomment the line "P = normalize..." in the forward_backward() function
            # set beta = 1,
            # then
            # running with ['abc d'] leads to the following alpha/beta values
            # alpha = array([1.0000, 0.3333, 0.5000, 1.0000, 1.0000, 1.0000])
            # beta = array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]) (this makes sense because normalization makes the lattice stochastic)
        else:
            original_lines = readlines(filename_train, rstrip=True)
            if filename_dev is not None and len(filename_dev) > 0:
                self.data.lines_dev = readlines(filename_dev, rstrip=True)
                original_lines += self.data.lines_dev

            self.data.lines = process_lines(original_lines, remove_whitespace, chop)

    def init_model(self):
        if self.params is None:
            return Model()
        else:
            exp_counts = Model(SegUtil.count_segments(self.data, self.params))
            # globals.most_common(exp_counts.Pseg, k=100)
            return self.M_step(exp_counts)

    def train(self):
        model = self.init_model()
        for iter in xrange(self.params.MAX_ITER):
            # E-step
            LL, exp_counts = self.E_step(model)
            print >> sys.stderr, "E-step", iter, " LL =", LL

            # M-step
            model = self.M_step(exp_counts)
            print >> sys.stderr, "M-step", iter

            if model.LL > LL:
                print >> sys.stderr, "Warning: log likelihood decreased!"
            if np.isclose(model.LL, LL, 1e-10):
                print >> sys.stderr, "Converged at iteration", iter
                break
            model.LL = LL

        self.model = model
        return model

    def M_step(self, ec):
        # compute the next model from the expected counts
        # by applying count-and-divide
        total = sum([ec.Pseg[seg] for seg in ec.Pseg])

        if total <= 0 or np.isnan(total):
            print >> sys.stderr, '## Warning: total is', total

        Pseg = {seg: (ec.Pseg[seg]/total) for seg in ec.Pseg}
        return Model(Pseg)


    def E_step(self, model):
        # initialize the expected counts
        Pseg = defaultdict(lambda: 0)
        LL = 0

        for i, line in enumerate(self.data.lines):       # go over each sentence and collect expected counts
            logprob_i, counts_i = MonotoneFSTUtil.forward_backward(model, self.params, line)
            LL += logprob_i

            for seg in counts_i:
                Pseg[seg] += counts_i[seg]      # accumulate the expected counts

        ec = Model(Pseg)
        return LL, ec

    def viterbi_decode(self, lines=None):
        if lines is None:
            lines = process_lines(self.data.lines_dev, self.params.remove_whitespace, self.params.chop)

        decodes = []
        for i, line in enumerate(lines):
            decode_i = MonotoneFSTUtil.viterbi(self.model, self.params, line, line_no=i)
            I = decode_i[0]
            decodes.append([SegUtil.segment_str(line, I)] + list(decode_i))
        return decodes

    def score(self):
        decodes = self.viterbi_decode()
        decoded_lines = [" ".join(line) for line, I, p in decodes]

        tp = fp = fn = 0
        for decoded, gold in zip(decoded_lines, self.data.lines_dev):
            correct, incorrect, missed = compare_whitespace(decoded, gold)
            tp += correct
            fp += incorrect
            fn += missed

        p = tp * 1.0 / (tp + fp)
        r = tp * 1.0 / (tp + fn)

        F = 2 * p * r / (p + r)
        return F, p, r

