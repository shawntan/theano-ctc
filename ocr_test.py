# coding=utf-8
import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import hinton
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import ctc
import font
import lstm
from ocr import *

if __name__ == "__main__":
    import sys
    test_word = sys.argv[1]

    P = Parameters()
    X = T.matrix('X')

    predict = build_model(P,8,512,len(font.chars)+1)
    probs = predict(X)
    test = theano.function(inputs=[X],outputs=probs)
    P.load('model.pkl')
    image = font.imagify(test_word)
    hinton.plot(image.astype(np.float32).T[::-1])
    y_seq = label_seq(test_word)
    probs = test(image)
    print " ", ' '.join(font.chars[i] if i < len(font.chars) else "_" for i in np.argmax(probs,axis=1))
    hinton.plot(probs[:,y_seq].T,max_arr=1.)

