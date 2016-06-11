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

def create_skip_mask(batched_labels):
    return T.neq(batched_labels[:,:-1],batched_labels[:,1:])

def extract_log_probs(log_probs, blanked_labels):
    batch_size, label_size = blanked_labels.shape
    sequence_length, batch_size, _ = log_probs.shape

    return log_probs[
        :,
        T.arange(batch_size).dimshuffle(0, 'x'),
        blanked_labels.dimshuffle(0, 1)
    ]


def recurrence(log_p_curr, log_p_prev, skip_mask=None):
    if skip_mask is None:
        skip_mask = T.ones_like(log_p_curr[:, 1:-2:2])


    # normalise and bring back to p space
    k = T.max(log_p_prev, axis=1, keepdims=True)
    norm_p_prev = T.switch(
        T.isinf(log_p_prev), 0, T.exp(log_p_prev - k))  # set -inf to 0

    # previous
    _result = norm_p_prev
    # add shift of previous
    _result = T.inc_subtensor(_result[:, 1:],   norm_p_prev[:, :-1])
    # add skips of previous
    _result = T.inc_subtensor(_result[:, 3::2],
            T.switch(skip_mask,norm_p_prev[:, 1:-2:2],0))
    # current
    # log(p) should be 0 for first 2 terms
    result = T.switch(
        T.eq(_result, 0),
        -np.inf,
        log_p_curr + T.log(_result) + k
    )
    return result


def forward_backward_pass(log_probs, label_mask, frame_mask, skip_mask=None):
    # log_probs:  time x batch_size x label_size
    # label_mask: batch_size x label_size
    # frame_mask: time x batch_size
    if skip_mask is None:
        skip_mask = T.ones_like(log_probs[0, :, 1:-2:2])

    time, batch_size, label_size = log_probs.shape
    start_idxs = label_size - T.sum(label_mask, axis=1)
    infs = T.alloc(-np.inf, batch_size, label_size)

    def forward_backward(f_mask, b_mask, f_curr, b_curr, f_prev, b_prev):
        f_next = T.switch(f_mask, recurrence(f_curr, f_prev, skip_mask), infs)
        b_next = T.switch(b_mask, recurrence(b_curr, b_prev, skip_mask[:,::-1]), b_prev)
        return f_next, b_next

    f_init_logp = T.set_subtensor(infs[:, 0], 0)
    b_init_logp = T.set_subtensor(infs[T.arange(batch_size), start_idxs], 0)
    f_mask_seq = frame_mask
    b_mask_seq = frame_mask[::-1]
    f_logp_seq = log_probs
    b_logp_seq = log_probs[::-1, :, ::-1]

    [f_acc, b_acc], _ = theano.scan(
        fn=forward_backward,
        sequences=[f_mask_seq, b_mask_seq, f_logp_seq, b_logp_seq],
        outputs_info=[f_init_logp, b_init_logp]
    )

    return f_acc + b_acc[::-1, :, ::-1] - log_probs


def acc_cost(log_probs, label_mask, frame_mask, skip_mask=None):
    seq_acc_logp = forward_backward_pass(
        log_probs,
        label_mask,
        frame_mask,
        skip_mask
    )
    k = T.max(seq_acc_logp, axis=2, keepdims=True)
    log_sum_p = T.log(T.sum(
        T.switch(T.isinf(seq_acc_logp), 0, T.exp(seq_acc_logp - k)),
        axis=2
    )) + k.dimshuffle(0, 1)
    return T.sum(log_sum_p, axis=0)


def cost(linear_out, frame_lengths, labels, label_lengths):
    log_probs = log_softmax(linear_out)
    blanked_labels = insert_blanks(labels)
    extracted_log_probs = extract_log_probs(log_probs, blanked_labels)
    blanked_labels_length = label_lengths * 2 + 1
    label_mask = T.arange(blanked_labels.shape[1]).dimshuffle('x', 0) <\
        blanked_labels_length.dimshuffle(0, 'x')
    frame_mask = T.arange(linear_out.shape[0]).dimshuffle(0, 'x') <\
        frame_lengths.dimshuffle('x', 0)
    frame_mask = frame_mask.dimshuffle(0, 1, 'x')
    return acc_cost(
        extracted_log_probs,
        label_mask,
        frame_mask,
        create_skip_mask(labels)
    )
