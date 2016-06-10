import unittest

import theano
import theano.tensor as T
import numpy as np

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import ctc


def recurrence(p_curr, p_prev):
    # add previous
    _result = p_prev
    # add shift of previous
    _result = T.inc_subtensor(_result[1:],   p_prev[:-1])
    # add skips of previous
    _result = T.inc_subtensor(_result[3::2], p_prev[1:-2:2])
    # current
    _result = _result * p_curr
    return _result


def gs_recurrence_pass(log_probs):
    probs = T.exp(log_probs)
    init_probs = T.alloc(np.float32(0), log_probs.shape[1])
    init_probs = T.set_subtensor(init_probs[0], np.float32(1))
    pass_probs, _ = theano.scan(
        fn=recurrence,
        sequences=[probs],
        outputs_info=[init_probs]
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


class CheckLabelsTestCase(CTCTestCase):

    def setUp(self):
        self.labels = T.as_tensor_variable(np.array([
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0, -1],
            [2, 1, 0, -1, -1],
            [1, 0, -1, -1, -1]
        ], dtype=np.int32))

        self.log_probs = ctc.log_softmax(T.as_tensor_variable(-np.array([
            [[0.,   0.1,   0.2,   0.3,   0.4,   0.5],
             [10.,  10.1,  10.2,  10.3,  10.4,  10.5],
             [20.,  20.1,  20.2,  20.3,  20.4,  20.5],
             [30.,  30.1,  30.2,  30.3,  30.4,  30.5]],
            [[1.,   1.1,   1.2,   1.3,   1.4,   1.5],
             [11.,  11.1,  11.2,  11.3,  11.4,  11.5],
             [21.,  21.1,  21.2,  21.3,  21.4,  21.5],
             [31.,  31.1,  31.2,  31.3,  31.4,  31.5]],
            [[2.,   2.1,   2.2,   2.3,   2.4,   2.5],
             [12.,  12.1,  12.2,  12.3,  12.4,  12.5],
             [22.,  22.1,  22.2,  22.3,  22.4,  22.5],
             [32.,  32.1,  32.2,  32.3,  32.4,  32.5]],
            [[3.,   3.1,   3.2,   3.3,   3.4,   3.5],
             [13.,  13.1,  13.2,  13.3,  13.4,  13.5],
             [23.,  23.1,  23.2,  23.3,  23.4,  23.5],
             [33.,  33.1,  33.2,  33.3,  33.4,  33.5]],
            [[4.,   4.1,   4.2,   4.3,   4.4,   4.5],
             [14.,  14.1,  14.2,  14.3,  14.4,  14.5],
             [24.,  24.1,  24.2,  24.3,  24.4,  24.5],
             [34.,  34.1,  34.2,  34.3,  34.4,  34.5]]
        ], dtype=np.float32)))

    def test_log_softmax(self):
        self.assertTrue(np.allclose(
            T.sum(T.exp(self.log_probs), axis=-1).eval(), 1))

    def test_labels(self):
        blanked_labels = np.array([
            [-1, 4, -1, 3, -1,  2, -1,  1, -1,  0, -1],
            [-1, 3, -1, 2, -1,  1, -1,  0, -1, -1, -1],
            [-1, 2, -1, 1, -1,  0, -1, -1, -1, -1, -1],
            [-1, 1, -1, 0, -1, -1, -1, -1, -1, -1, -1]
        ], dtype=np.int32)

        self.assertTrue(
            (ctc.insert_blanks(self.labels).eval() == blanked_labels).all())

    def test_extract_labels(self):
        out = ctc.extract_log_probs(
            self.log_probs,
            ctc.insert_blanks(self.labels)
        ).eval()
        self.assertTrue(np.allclose(
            self.log_probs[0, 0, self.labels[0, 0]].eval(),
            out[0, 0, 1]
        ))
        self.assertTrue(np.allclose(
            self.log_probs[0, 1, self.labels[1, 0]].eval(),
            out[0, 1, 1]
        ))
        self.assertTrue(np.allclose(
            self.log_probs[2, 1, self.labels[1, 0]].eval(),
            out[2, 1, 1]
        ))
        self.assertTrue(np.allclose(
            self.log_probs[0, 0, self.labels[0, 1]].eval(),
            out[0, 0, 3]
        ))
        self.assertTrue(np.allclose(
            self.log_probs[0, 1, self.labels[1, 1]].eval(),
            out[0, 1, 3]
        ))
        self.assertTrue(np.allclose(
            self.log_probs[2, 1, self.labels[1, 1]].eval(),
            out[2, 1, 3]
        ))


class CheckRecurrenceCorrectnessTestCase(CTCTestCase):

    def test_recurrence_correctness(self):
        gs_output = gs_recurrence_pass(self.log_probs).eval(self.subs)
        ctc_output = ctc.recurrence_pass(
            T.cast(T.zeros((1,)), 'int32'),
            self.log_probs.dimshuffle(0, 'x', 1)
        )[:, 0, :].eval(self.subs)
        compare_idxs = ~np.isinf(gs_output)
        self.assertTrue(np.allclose(
            gs_output[compare_idxs], ctc_output[compare_idxs]))

    def test_recurrence_differentiable(self):
        subs = {self.X: np.random.randn(10, 10).astype(np.float32)}
        ctc_output = ctc.recurrence_pass(
            T.cast(T.zeros((2,)), 'int32'),
            self.log_probs.reshape(
                (self.log_probs.shape[0] / 2, 2, self.log_probs.shape[1])))
        cost = -T.mean(ctc_output[-1, :, -1])
        [g] = T.grad(cost, wrt=[self.W_test])
        self.assertTrue((~np.isnan(g.eval(self.subs))).all())

    def test_recurrence_with_offset(self):
        actual_input = np.random.randn(20, 10).astype(np.float32)
        subs = {self.X: actual_input}
        sample_input = self.log_probs.reshape(
            (self.log_probs.shape[0] / 2, 2, self.log_probs.shape[1]))
        sample_padding = T.alloc(-np.inf, self.log_probs.shape[0] / 2, 2, 2)
        sample_input_padded = T.concatenate(
            [sample_padding, sample_input], axis=-1)
        ctc_output = ctc.recurrence_pass(
            T.cast(2 * T.ones((2,)), 'int32'),
            sample_input_padded
        )
        gs_output = gs_recurrence_pass(sample_input[:, 0, :])
        gs_vals = gs_output.eval(subs)
        ctc_vals = ctc_output.eval(subs)[:, 0, 2:]
        compare_idxs = ~np.isinf(gs_vals)
        self.assertTrue(np.allclose(
            gs_vals[compare_idxs], ctc_vals[compare_idxs]))

if __name__ == "__main__":
    unittest.main()
