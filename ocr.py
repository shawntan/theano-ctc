# coding=utf-8
import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import hinton
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import ctc1
import ctc
import font


def build_rnn(hidden_inputs, W_hidden_hidden, b_hidden,
              initial_hidden):
    def step(input_curr, hidden_prev):
        hidden = T.tanh(
            T.dot(hidden_prev, W_hidden_hidden) +
            input_curr +
            b_hidden
        )
        return hidden
    hidden, _ = theano.scan(
        step,
        sequences=[hidden_inputs],
        outputs_info=[initial_hidden]
    )
    return hidden


def build_model(P, X, input_size, hidden_size, output_size):
    P.W_input_hidden =  U.initial_weights(input_size, hidden_size)
    P.W_hidden_hidden = U.initial_weights(hidden_size, hidden_size)
    P.W_hidden_output = U.initial_weights(hidden_size, output_size)
    P.b_hidden = U.initial_weights(hidden_size)
    P.i_hidden = U.initial_weights(hidden_size)
    P.b_output = U.initial_weights(output_size)
    hidden = build_rnn(T.dot(X, P.W_input_hidden),
                       P.W_hidden_hidden, P.b_hidden, P.i_hidden)

    predict = T.nnet.softmax(T.dot(hidden, P.W_hidden_output) + P.b_output)

    return X, predict


def label_seq(string):
    idxs = font.indexify(string)
    result = np.ones((len(idxs) * 2 + 1,), dtype=np.int32) * -1
    result[np.arange(len(idxs)) * 2 + 1] = idxs
    return result


if __name__ == "__main__":
    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')
    X,predict = build_model(P,X,8,256,len(font.chars)+1)

    cost = ctc1.cost(predict, Y)
    params = P.values()
    gradients = T.grad(cost, wrt=params)
    train = theano.function(
        inputs=[X, Y],
        outputs=cost,
        updates = [ (p,p-1e-6 * g) for p,g in zip(params,gradients) ]
    )
    test = theano.function(
            inputs=[X,Y],
            outputs=predict[:,Y] / T.sum(predict[:,Y],axis=1).dimshuffle(0,'x')
        )
    training_examples = "the quick brown fox jumps over the lazy dog".split()
    import random
    for _ in xrange(1000):
        random.shuffle(training_examples)
        for string in training_examples:
            print train(font.imagify(string),label_seq(string))
        hinton.plot(test(font.imagify("the"),label_seq("the")).T,max_arr=1)
        hinton.plot(font.imagify("the").T[::-1].astype('float32'))



