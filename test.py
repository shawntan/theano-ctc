import unittest

import theano
import theano.tensor as T
import numpy as np

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import ctc

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

def gs_recurrence_pass(log_probs):
    probs = T.exp(log_probs)
    init_probs = T.alloc(np.float32(0),log_probs.shape[1])
    init_probs = T.set_subtensor(init_probs[0],np.float32(1))
    [_, pass_probs], _ = theano.scan(
            fn=recurrence,
            sequences=[probs],
            outputs_info=[np.int32(0),init_probs]
        )
    return T.log(pass_probs)



class CheckRecurrenceCorrectness(unittest.TestCase):
    def setUp(self):
        self.input = T.log(T.nnet.softmax(np.random.randn(10,5).astype(np.float32)))

    def test_correctness(self):
        gs_output = gs_recurrence_pass(self.input).eval()
        ctc_output = ctc.recurrence_pass(self.input.dimshuffle(0,'x',1)).eval()[:,0,:]
        compare_idxs = ~np.isinf(gs_output)
        self.assertTrue(np.allclose(gs_output[compare_idxs],ctc_output[compare_idxs]))



if __name__ == "__main__":
    unittest.main()
