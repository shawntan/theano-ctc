import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

eps = 1e-40

def recurrence_relation(size):
    big_I = T.eye(size+2)
    return T.eye(size) + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast(T.arange(size) % 2,'float32')

def path_probs(predict, Y):
    P = predict[:,Y]
    rr = recurrence_relation(Y.shape[0])
    def step(p_curr,p_prev):
        result = p_curr * T.dot(p_prev,rr)
        return result
    probs,_ = theano.scan(
            step,
            sequences = [P],
            outputs_info = [T.eye(Y.shape[0])[0]]
        )
    return probs

def cost(predict, Y):
    forward_probs  = path_probs(predict,Y)
    backward_probs = path_probs(predict[::-1],Y[::-1])[::-1,::-1]
    probs = forward_probs * backward_probs / predict[:,Y]
    total_prob = T.sum(probs)
    return -T.log(total_prob)
