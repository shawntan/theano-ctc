import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print


def log_softmax(X):
    k = T.max(X, axis=-1, keepdims=True)
    norm_X = X - k
    log_sum_exp_X = T.log(T.sum(T.exp(norm_X), axis=-1, keepdims=True))
    return norm_X - log_sum_exp_X


def insert_blanks(batched_labels):
    _result = T.alloc(
        -1,
        batched_labels.shape[0],
        batched_labels.shape[1] * 2 + 1
    )
    _result = T.set_subtensor(_result[:, 1:-1:2], batched_labels)
    return _result


def extract_log_probs(log_probs, blanked_labels):
    batch_size, label_size = blanked_labels.shape
    sequence_length, batch_size, _ = log_probs.shape

    return log_probs[
        :,
        T.arange(batch_size).dimshuffle(0, 'x'),
        blanked_labels.dimshuffle(0, 1)
    ]


def recurrence(log_p_curr, log_p_prev):
    # normalise and bring back to p space
    k = T.max(log_p_prev, axis=1, keepdims=True)
    norm_p_prev = T.switch(
        T.isinf(log_p_prev), 0, T.exp(log_p_prev - k))  # set -inf to 0

    # previous
    _result = norm_p_prev
    # add shift of previous
    _result = T.inc_subtensor(_result[:, 1:],   norm_p_prev[:, :-1])
    # add skips of previous
    _result = T.inc_subtensor(_result[:, 3::2], norm_p_prev[:, 1:-2:2])
    # current
    # log(p) should be 0 for first 2 terms
    result = T.switch(
        T.eq(_result, 0),
        -np.inf,
        log_p_curr + T.log(_result) + k
    )
    return result

def forward_backward_pass(log_probs, label_mask, frame_mask):
    # log_probs:  time x batch_size x label_size
    # label_mask: batch_size x label_size
    # frame_mask: time x batch_size
    time, batch_size, label_size = log_probs.shape
    start_idxs = label_size - T.sum(label_mask,axis=1)
    init_infs = T.alloc(-np.inf, batch_size, label_size)

    def forward_backward(f_mask, b_mask, f_curr, b_curr, f_prev, b_prev):
        f_next = T.switch(f_mask, recurrence(f_curr, f_prev), f_prev)
        b_next = T.switch(b_mask, recurrence(b_curr, b_prev), b_prev)
        return f_next, b_next

    f_init_logp = T.set_subtensor(init_infs[:,0], 0)
    b_init_logp = T.set_subtensor(init_infs[T.arange(batch_size),start_idxs], 0)
    f_mask_seq = frame_mask
    b_mask_seq = frame_mask[::-1]
    f_logp_seq = log_probs
    b_logp_seq = log_probs[::-1,:,::-1]

    [f_acc, b_acc], _ = theano.scan(
        fn=forward_backward,
        sequences=[f_mask_seq, b_mask_seq, f_logp_seq, b_logp_seq],
        outputs_info=[f_init_logp, b_init_logp]
    )

    return f_acc + b_acc[::-1,:,::-1] - log_probs




