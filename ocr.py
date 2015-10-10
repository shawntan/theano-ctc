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
import lstm

def build_model(P, input_size, hidden_size, output_size):
    lstm_layer = lstm.build(P,"lstm",input_size,hidden_size)
    P.W_output = np.zeros((hidden_size,output_size))
    P.b_output = np.zeros((output_size,))

    def model(X):
        hidden = lstm_layer(X)[1]
        return T.nnet.softmax(T.dot(hidden,P.W_output) + P.b_output)
    return model


def label_seq(string):
    idxs = font.indexify(string)
    result = np.ones((len(idxs) * 2 + 1,), dtype=np.int32) * -1
    result[np.arange(len(idxs)) * 2 + 1] = idxs
    return result


if __name__ == "__main__":
    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')
    predict = build_model(P,8,256,len(font.chars)+1)


    probs = predict(X)
    alpha = 0.1
    smoothed_predict = (1-alpha) * probs + alpha * 1./(len(font.chars)+1)
    cost = ctc1.cost(smoothed_predict, Y)
    params = P.values()
    gradients = T.grad(cost, wrt=params)
    
    gradient_acc = [ 
            theano.shared(0 * p.get_value()) for p in params 
        ]
    counter = theano.shared(np.float32(0.))
    acc = theano.function(
        inputs=[X, Y],
        outputs=cost,
        updates = [
            (a,a + g) for a,g in zip(gradient_acc,gradients)
        ] + [(counter,counter + np.float32(1.))]
    )
    update = theano.function(
            inputs=[],outputs=[],
            updates = updates.rmsprop(
                params,[ g / counter for g in gradient_acc ],
                learning_rate=1e-5
            ) + [ (counter,np.float32(0.))]
        )

    test = theano.function(
         inputs=[X,Y],
         outputs=probs[:,Y] / T.sum(probs[:,Y],axis=1).dimshuffle(0,'x')
        )

    training_examples = [ word.strip() for word in open('dictionary.txt') ]
    import random
    for _ in xrange(1500):
        random.shuffle(training_examples)
        for string in training_examples[:100]:
            print acc(font.imagify(string),label_seq(string))
        update()
        hinton.plot(test(font.imagify("test"),label_seq("test")).T)
        hinton.plot(font.imagify("test").T[::-1].astype('float32'))



