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


def build_model(P, input_size, hidden_size, output_size):
    lstm_layer = lstm.build(P, "lstm", input_size, hidden_size)
    P.W_output = np.zeros((hidden_size, output_size))
    P.b_output = np.zeros((output_size,))

    def model(X):
        hidden = lstm_layer(X)[1]
        return T.nnet.softmax(T.dot(hidden, P.W_output) + P.b_output)
    return model


def label_seq(string):
    idxs = font.indexify(string)
    return idxs


if __name__ == "__main__":
    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')

    predict = build_model(P, 8, 512, len(font.chars) + 1)

    probs = predict(X)
    alpha = 0.5
    params = P.values()
    cost = ctc.cost(probs, Y)  # + 1e-8 * sum(T.sum(T.sqr(w)) for w in params)
    gradients = T.grad(cost, wrt=params)

    gradient_acc = [theano.shared(0 * p.get_value()) for p in params]
    counter = theano.shared(np.float32(0.))
    acc = theano.function(
        inputs=[X, Y],
        outputs=cost,
        updates=[
            (a, a + g) for a, g in zip(gradient_acc, gradients)
        ] + [(counter, counter + np.float32(1.))]
    )
    update = theano.function(
        inputs=[], outputs=[],
        updates=updates.momentum(
            params, [g / counter for g in gradient_acc],
        ) + [(a, np.float32(0) * a) for a in gradient_acc] + [(counter, np.float32(0.))]
    )

    test = theano.function(
        inputs=[X, Y],
        outputs=probs[:, Y]
    )
    training_examples = [word.strip() for word in open('dictionary.txt')]
    import random
    for _ in xrange(1500):
        random.shuffle(training_examples)
        for i, string in enumerate(training_examples):
            print acc(font.imagify(string), label_seq(string))
            if i % 20 == 0:
                update()
            if i % 100 == 0:
                hinton.plot(test(font.imagify("test"),
                                 label_seq("test")).T, max_val=1.)
                hinton.plot(font.imagify("test").T[::-1].astype('float32'))
        P.save('model.pkl')
