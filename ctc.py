import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print


def update_log_p(skip_idxs, zeros, active, log_p_curr, log_p_prev):
    active_skip_idxs = skip_idxs[(skip_idxs < active).nonzero()]
    active_next = T.cast(T.minimum(
        T.maximum(
            active + 1,
            T.max(T.concatenate([active_skip_idxs, [-1]])) + 2 + 1
        ),
        log_p_curr.shape[0]
    ), 'int32')

    common_factor = T.max(log_p_prev[:active])
    p_prev = T.exp(log_p_prev[:active] - common_factor)
    _p_prev = zeros[:active_next]
    # copy over
    _p_prev = T.set_subtensor(_p_prev[:active], p_prev)
    # previous transitions
    _p_prev = T.inc_subtensor(_p_prev[1:], _p_prev[:-1])
    # skip transitions
    _p_prev = T.inc_subtensor(
        _p_prev[active_skip_idxs + 2], p_prev[active_skip_idxs])
    updated_log_p_prev = T.log(_p_prev) + common_factor

    log_p_next = T.set_subtensor(
        zeros[:active_next],
        log_p_curr[:active_next] + updated_log_p_prev
    )
    return active_next, log_p_next


def create_skip_idxs(Y):
    skip_idxs = T.arange((Y.shape[0] - 3) // 2) * 2 + 1

    non_repeats = T.neq(Y[skip_idxs], Y[skip_idxs + 2])
    return skip_idxs[non_repeats.nonzero()]


def path_probs(predict, Y, alpha=1e-4):
    smoothed_predict = (1 - alpha) * \
        predict[:, Y] + alpha * np.float32(1.) / Y.shape[0]
    L = T.log(smoothed_predict)
    zeros = T.zeros_like(L[0])
    base = T.set_subtensor(zeros[:1], np.float32(1))
    log_first = zeros

    f_skip_idxs = create_skip_idxs(Y)
    # there should be a shortcut to calculating this
    b_skip_idxs = create_skip_idxs(Y[::-1])

    def step(log_f_curr, log_b_curr, f_active, log_f_prev, b_active, log_b_prev):
        f_active_next, log_f_next = update_log_p(
            f_skip_idxs, zeros, f_active, log_f_curr, log_f_prev)
        b_active_next, log_b_next = update_log_p(
            b_skip_idxs, zeros, b_active, log_b_curr, log_b_prev)
        return f_active_next, log_f_next, b_active_next, log_b_next
    [f_active, log_f_probs, b_active, log_b_probs], _ = theano.scan(
        step,
        sequences=[
            L,
            L[::-1, ::-1]
        ],
        outputs_info=[
            np.int32(1), log_first,
            np.int32(1), log_first,
        ]
    )
    idxs = T.arange(L.shape[1]).dimshuffle('x', 0)
    mask = (idxs < f_active.dimshuffle(0, 'x')) & (
        idxs < b_active.dimshuffle(0, 'x'))[::-1, ::-1]
    log_probs = log_f_probs + log_b_probs[::-1, ::-1] - L
    return log_probs, mask


def cost(predict, Y):
    log_probs, mask = path_probs(predict, Y)
    common_factor = T.max(log_probs)
    total_log_prob = T.log(
        T.sum(T.exp(log_probs - common_factor)[mask.nonzero()])) + common_factor
    return -total_log_prob


if __name__ == "__main__":
    import ctc_old
    probs = T.nnet.softmax(np.random.randn(20, 11).astype(np.float32))
    labels = theano.shared(np.array(
        [-1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1], dtype=np.int32))

    print ctc_old.cost(probs, labels).eval()
    print cost(probs, labels).eval()
