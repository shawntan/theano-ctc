import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print

def recurrence_pass(log_probs):
    start_idx = T.cast(T.zeros((log_probs.shape[1],)),'int32')
    log_init_probs = T.alloc(-np.inf,log_probs.shape[1],log_probs.shape[2])
    log_init_probs = T.set_subtensor(log_init_probs[start_idx,0],0)
    [_,pass_log_probs], _ = theano.scan(
            fn=recurrence,
            sequences=[log_probs],
            outputs_info=[start_idx, log_init_probs]
        )
    return pass_log_probs

def recurrence(log_p_curr, t, log_p_prev):
    idx = T.arange(log_p_prev.shape[1])
#    mask = idx.dimshuffle('x',0) < t.dimshuffle(0,'x')

    # normalise and bring back to p space
    k = T.max(log_p_prev, axis=1,keepdims=True)
    norm_p_prev = T.exp(log_p_prev - k)
    norm_p_prev = T.switch(T.isinf(log_p_prev), 0, norm_p_prev) # set -inf to 0

    # previous
    _result = norm_p_prev
    # add shift of previous
    _result = T.inc_subtensor(_result[:,1:],   norm_p_prev[:,:-1])
    # add skips of previous
    _result = T.inc_subtensor(_result[:,3::2], norm_p_prev[:,1:-2:2])
    log_p_transitions = T.switch(T.eq(_result,0), -np.inf, T.log(_result) + k)
    # current
    # log(p) should be 0 for first 2 terms
    _result = log_p_curr + log_p_transitions

    t = t + 2
    result = _result
    return t, result

