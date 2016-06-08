import unittest

import theano
import theano.tensor as T
import numpy as np

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import ctc


def recurrence(p_curr, t, p_prev):
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
    init_probs = T.alloc(np.float32(0), log_probs.shape[1])
    init_probs = T.set_subtensor(init_probs[0], np.float32(1))
    [_, pass_probs], _ = theano.scan(
        fn=recurrence,
        sequences=[probs],
        outputs_info=[np.int32(0), init_probs]
    )
    return T.log(pass_probs)


class CTCTestCase(unittest.TestCase):

    def setUp(self):
        label_length = 5
        labels = np.empty((label_length * 2 + 1,), dtype=np.int32)
        labels[1::2] = np.arange(label_length)
        labels[::2] = -1

        input_size = 10
        predict_size = 20
        P = Parameters()
        P.W_test = np.zeros((input_size, predict_size))
        self.W_test = P.W_test
        self.X = T.matrix('X')
        probs = T.nnet.softmax(T.dot(self.X, self.W_test))
        self.log_probs = T.log(probs[:, labels])
        self.subs = {self.X: np.random.randn(10, 10).astype(np.float32)}


class CheckRecurrenceCorrectnessTestCase(CTCTestCase):

    def test_recurrence_correctness(self):
        gs_output = gs_recurrence_pass(self.log_probs).eval(self.subs)
        ctc_output = ctc.recurrence_pass(
            self.log_probs.dimshuffle(0, 'x', 1)
        )[:, 0, :].eval(self.subs)
        compare_idxs = ~np.isinf(gs_output)
        self.assertTrue(np.allclose(
            gs_output[compare_idxs], ctc_output[compare_idxs]))

    def test_recurrence_differentiable(self):
        subs = {self.X: np.random.randn(10, 10).astype(np.float32)}
        ctc_output = ctc.recurrence_pass(self.log_probs.reshape(
            (self.log_probs.shape[0] / 2, 2, self.log_probs.shape[1])))
        cost = -T.mean(ctc_output[-1, :, -1])
        [g] = T.grad(cost, wrt=[self.W_test])
        self.assertTrue((~np.isnan(g.eval(self.subs))).all())


if __name__ == "__main__":
    unittest.main()
