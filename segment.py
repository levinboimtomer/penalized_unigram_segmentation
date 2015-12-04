__author__ = 'tomerlevinboim'
import sys
import argparse
import Segmenter


def parseCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", default="", help="Train filename")
    parser.add_argument("-d", "--dev", dest="dev", default="", help="development filename")
    parser.add_argument("-b", "--beta", dest="beta", default=1.3, type=float, help="Higher beta makes longer sequences less likely")
    parser.add_argument("-c", "--offset", dest="offset", default=0, type=float, help="Set c higher than 0 if short segments are unlikely")
    parser.add_argument("-S", "--maxseglength", dest="MAX_SEG_LENGTH", default=10, type=int, help="maximum segment length")
    parser.add_argument("-i", "--iterations", dest="ITERATIONS", default=6, type=int, help="EM iterations")

    opts = parser.parse_args()
    return opts, parser


if __name__ == '__main__':
    opts, parser = parseCommandLine()                                               # initialize
    if opts.train == "":
        parser.print_help()
    else:
        segmenter = Segmenter.Segmenter(opts)
        segmenter.loadData(opts.train, opts.dev, remove_whitespace=True)

        model = segmenter.train()                                               # train
        decodes = segmenter.viterbi_decode(segmenter.data.lines)                # decode
        if opts.dev != "":
            F, p, r = segmenter.score()
            print >> sys.stderr, "beta =", opts.beta, "==> F = %2.1f with p@r = %2.1f@%2.1f" % (F * 100, p * 100, r * 100)

        for (line, indices, p) in decodes:                                      # output
            print " ".join(line)

