__author__ = 'tomerlevinboim'

import Segmenter



if __name__ == '__main__':
    filename_train = 'data/train.1K.en'
    filename_dev = 'data/dev.en'
    #beta = float(sys.argv[1])#
    #filename = 'test'

    for beta in [1.31]:
        segmenter = Segmenter.Segmenter(beta, MAX_SEG_LENGTH=10, EM_ITER=4, beta_offset=3)
        segmenter.loadData(filename_train, filename_dev, remove_whitespace=True)
        model = segmenter.train()
        #decodes = segmenter.viterbi_decode()
        F, p, r = segmenter.score()
        print "beta = ", beta, "F = %2.1f with p@r = %2.1f@%2.1f" % (F * 100, p * 100, r * 100)
