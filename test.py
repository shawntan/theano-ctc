import unittest
import theano
import theano.tensor as T
import numpy as np
import ctc


def gs_recurrence(p_curr, p_prev):
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
        fn=gs_recurrence,
        sequences=[probs],
        outputs_info=[init_probs]
    )
    return T.log(pass_probs)


def ctc_recurrence_pass(start_idx, log_probs):
    log_init_probs = T.alloc(-np.inf, log_probs.shape[1], log_probs.shape[2])
    log_init_probs = T.set_subtensor(
        log_init_probs[T.arange(log_init_probs.shape[0]), start_idx], 0)
    pass_log_probs, _ = theano.scan(
        fn=ctc.recurrence,
        sequences=[log_probs],
        outputs_info=[log_init_probs]
    )
    return pass_log_probs


class CTCTestCase(unittest.TestCase):

    def setUp(self):
        self.labels = T.as_tensor_variable(np.array([
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0, -1],
            [2, 1, 0, -1, -1],
            [1, 0, -1, -1, -1]
        ], dtype=np.int32))

        self.labels_length = T.as_tensor_variable(np.array(
            [5, 4, 3, 2], dtype=np.int32
        ))

        self.data = T.as_tensor_variable(
            np.random.randn(10, 4, 5).astype(np.float32)
        )

        self.data_length = T.as_tensor_variable(np.array(
            [10, 10, 10, 10], dtype=np.int32
        ))

        self.transform = theano.shared(
            np.random.randn(5, 6).astype(np.float32)
        )
        self.lin_output = T.dot(self.data, self.transform)
        self.log_probs = ctc.log_softmax(self.lin_output)

        self.blanked_labels = ctc.insert_blanks(self.labels)

        self.extracted_log_probs = ctc.extract_log_probs(
            self.log_probs,
            self.blanked_labels
        )


class CheckLabelsTestCase(CTCTestCase):

    def test_log_softmax(self):
        self.assertTrue(np.allclose(
            T.sum(T.exp(self.log_probs), axis=-1).eval(), 1))

    def test_blank_insertion(self):
        blanked_labels = np.array([
            [-1, 4, -1, 3, -1,  2, -1,  1, -1,  0, -1],
            [-1, 3, -1, 2, -1,  1, -1,  0, -1, -1, -1],
            [-1, 2, -1, 1, -1,  0, -1, -1, -1, -1, -1],
            [-1, 1, -1, 0, -1, -1, -1, -1, -1, -1, -1]
        ], dtype=np.int32)

        self.assertTrue(
            (self.blanked_labels.eval() == blanked_labels).all())

    def test_extract_labels(self):
        out = self.extracted_log_probs.eval()
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
        gs_output = gs_recurrence_pass(self.log_probs[:, 0, :]).eval()
        ctc_output = ctc_recurrence_pass(
            T.cast(T.zeros((self.log_probs.shape[1],)), 'int32'),
            self.log_probs
        ).eval()[:, 0, :]
        compare_idxs = ~np.isinf(gs_output)
        self.assertTrue(np.allclose(
            gs_output[compare_idxs], ctc_output[compare_idxs]))

    def test_recurrence_differentiable(self):
        ctc_output = ctc_recurrence_pass(
            T.cast(T.zeros((self.log_probs.shape[1],)), 'int32'),
            self.extracted_log_probs
        )

        costs = -ctc_output[
            -1,
            T.arange(ctc_output.shape[1]),
            self.labels_length * 2
        ]

        [g] = T.grad(T.mean(costs), wrt=[self.transform])
        self.assertTrue((~np.isnan(g.eval())).all())

    def test_recurrence_with_offset(self):
        blanked_labels_length = self.labels_length * 2 + 1
        label_mask = T.arange(self.blanked_labels.shape[1]).dimshuffle('x', 0) <\
            blanked_labels_length.dimshuffle(0, 'x')
        reversed_log_probs = T.switch(
            label_mask, self.extracted_log_probs, -np.inf)[:, :, ::-1]
        offsets = self.blanked_labels.shape[1] - blanked_labels_length
        ctc_output = ctc_recurrence_pass(offsets, reversed_log_probs).eval()

        for i in xrange(4):

            off = offsets[i].eval()
            log_prob = reversed_log_probs[:, i, off:]
            gs_output = gs_recurrence_pass(log_prob).eval()
            compare_idxs = ~np.isinf(gs_output)

            self.assertTrue(np.allclose(
                ctc_output[:, i, off:][compare_idxs],
                gs_output[compare_idxs]
            ))


class CTCForwardBackwardTestCase(CTCTestCase):

    def test_ctc_backward_forward(self):
        blanked_labels_length = self.labels_length * 2 + 1
        label_mask = T.arange(self.blanked_labels.shape[1]).dimshuffle('x', 0) <\
            blanked_labels_length.dimshuffle(0, 'x')
        frame_mask = T.ones_like(self.extracted_log_probs)
        ctc_output = ctc.forward_backward_pass(
            self.extracted_log_probs,
            label_mask, frame_mask
        ).eval()

        for i in xrange(4):
            end = blanked_labels_length[i].eval()
            logp = self.extracted_log_probs[:, i, :end]
            f_pass = gs_recurrence_pass(logp)
            b_pass = gs_recurrence_pass(logp[::-1, ::-1])
            gs_output = (f_pass + b_pass[::-1, ::-1] - logp).eval()
            compare_idxs = ~np.isinf(gs_output)
            self.assertTrue(np.allclose(
                ctc_output[:, i, :end][compare_idxs],
                gs_output[compare_idxs]
            ))

    def test_ctc_differentiable(self):
        costs = ctc.cost(
            linear_out=self.lin_output,
            frame_lengths=self.data_length,
            labels=self.labels,
            label_lengths=self.labels_length
        )

        g = T.grad(T.mean(costs), wrt=self.transform)
        self.assertTrue((~np.isnan(g.eval())).all())

if __name__ == "__main__":
    unittest.main()
