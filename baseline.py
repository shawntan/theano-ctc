import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

from theano_toolkit.parameters import Parameters
P = Parameters()
P.W_test = np.random.randn(5,5)

W_test = P.W_test

def build_baseline():

    def recurrence(p_curr,t,p_prev):
        t = t + 2
        idxs = T.arange(p_prev.shape[0])

        # add previous
        _result = p_prev
        # add shift of previous
        _result = T.inc_subtensor(_result[1:],   p_prev[:-1])
        # add skips of previous
        _result = T.inc_subtensor(_result[3::2], p_prev[1:-2:2])

        # current
        _result = _result * p_curr

        #_result = T.set_subtensor(_result[t:],0)
        return t, _result

    X = T.matrix('X')
    probs = T.nnet.softmax(T.dot(X,P.W_test))
    init_probs = T.alloc(np.float32(0),X.shape[1])
    init_probs = T.set_subtensor(init_probs[0],np.float32(1))
    forward, _ = theano.scan(
            fn=recurrence,
            sequences=[probs],
            outputs_info=[np.int32(0),init_probs]
        )

    f = theano.function(inputs=[X],outputs=T.log(forward[-1]))
    return f
baseline = build_baseline()


