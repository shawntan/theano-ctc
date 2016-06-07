import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

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
    non_inf = idx < t

    return t, result, non_inf


if __name__ == "__main__":
    import baseline
    base = baseline.build_baseline()
    np.set_printoptions(precision=2)
    def log_space(inputs):
        X = T.matrix('X')

        probs = T.nnet.softmax(T.dot(X,baseline.W_test))
        probs = probs.dimshuffle(0, 'x', 1)

        log_init_probs = T.zeros((probs.shape[1],probs.shape[2]))
        log_forward, _ = theano.scan(
                fn=recurrence,
                sequences=[T.log(probs)],
                outputs_info=[T.zeros((probs.shape[1],)), log_init_probs, None]
            )
        cost = log_forward[1]
#        grads = T.grad(-cost[0,-1,-1],wrt=[baseline.W_test])
        f = theano.function(inputs=[X],outputs=[cost,log_forward[2]])
        return f(inputs)

    inputs = np.random.randn(10,5).astype(np.float32)
    orig = base(inputs)
    log_space, non_inf = log_space(inputs)
    print log_space[:,0,:] - orig
    print non_inf

