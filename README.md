A word segmentation model solves the problem of splitting tokens such as "thisisacompound" to their components ["this", "is", "a", "compound"].
This project implements an unsupervised (correct segmentations are not provided to the model) penalized unigram segmentation model.
The implementation is a slight variation on the model proposed by Liang et al. (2009) in their paper "Online EM for Unsupervised Models" in that it further introduces a new offset hyperparameter c (explained below) and also treats spaces as fixed word boundaries.

## Penalized Unigram Segmentation
The unigram segmentation model of a sentence s is defined as:

P(s) = \prod_k P[w_k]

where w_k denotes the k'th segment of s.
This model is extermely simple but is known to have a degenerate solution -- one that does not segment the sentence s at all.

This project is a (Python) implementation of a **penalized** unigram segmentation which avoids the degenerate solution.
According to this mode:

  P(s) \propto \prod_k P[w_k] * e^( abs(|w_k| - c))^\beta )

where:
* |w_k| is the length of the k'th segment w_k
* c is an offset parameter. Set c higher than 0 if short segments are unlikely.
* beta is a coefficient governing how unlikely are long segments.


### Testing and running the code

* To test the code run `python test.py --offset 2`. 
  * This should output F = 74.0 with p@r = 77.0@71.2 for beta=1.3
  * setting the offset to 0 (as in Liang et al.) results with F = 71.4 with p@r = 70.8@72.0 for beta=1.4
* To train on your own data, run `python segment.py --train filename --beta 1.3 > segmented.txt`. 
Omit the filename to see the usage message.

