import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

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
    W_input_hidden = U.create_shared(
        U.initial_weights(input_size, hidden_size))
    W_hidden_hidden = U.create_shared(
        U.initial_weights(hidden_size, hidden_size))
    W_hidden_output = U.create_shared(
        U.initial_weights(hidden_size, output_size))
    b_hidden = U.create_shared(U.initial_weights(hidden_size))
    i_hidden = U.create_shared(U.initial_weights(hidden_size))
    b_output = U.create_shared(U.initial_weights(output_size))
    hidden = build_rnn(T.dot(X, W_input_hidden),
                       W_hidden_hidden, b_hidden, i_hidden)

    predict = T.nnet.softmax(T.dot(hidden, W_hidden_output) + b_output)

    return X, predict


def label_seq(string):
    idxs = font.indexify(string)
    result = np.ones((len(idxs) * 2 + 1,), dtype=np.int32) * -1
    result[np.arange(len(idxs)) * 2 + 1] = idxs
    print result
    return result


if __name__ == "__main__":
    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')
    X, predict = build_model(P, X, 10, 10, 10)

    cost = ctc.cost(predict, Y)
    params = P.values()
    grad = T.grad(cost, wrt=params)
    train = theano.function(
        inputs=[X, Y],
        outputs=cost,
        updates=updates.adadelta(params, grad)
    )

    for _ in xrange(10):
        print train(np.eye(10, dtype=np.float32)[::-1], np.arange(10, dtype=np.int32))
