import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

def recurrence_pass(log_probs):
    log_init_probs = T.zeros((log_probs.shape[1],log_probs.shape[2]))
    [_,pass_log_probs], _ = theano.scan(
            fn=recurrence,
            sequences=[log_probs],
            outputs_info=[T.zeros((log_probs.shape[1],)), log_init_probs]
        )
    return pass_log_probs

def recurrence(log_p_curr, t, log_p_prev):
    idx = T.arange(log_p_prev.shape[1])
    mask = idx.dimshuffle('x',0) < t.dimshuffle(0,'x')

    # normalise and bring back to p space
    k = T.maximum(T.max(T.switch(mask,log_p_prev,-np.inf), axis=1), 0)
    norm_p_prev = T.exp(log_p_prev - k)
    norm_p_prev = T.switch(mask, norm_p_prev, 0) # set -inf to 0

    # previous
    _result = norm_p_prev
    # add shift of previous
    _result = T.inc_subtensor(_result[:,1:],   norm_p_prev[:,:-1])
    # add skips of previous
    _result = T.inc_subtensor(_result[:,3::2], norm_p_prev[:,1:-2:2])

    # current
    # log(p) should be 0 for first 2 terms
    _result = T.switch((t > 0).dimshuffle(0,'x'),
                log_p_curr + T.log(_result) + k,
                log_p_curr
            )

    t = t + 2
    result = _result #T.set_subtensor(_result[:,t:], np.float32(0))
    return t, result

