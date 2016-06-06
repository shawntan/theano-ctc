import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

def recurrence(log_p_curr, t, log_p_prev):
    idx = T.arange(log_p_prev.shape[0])
    k = T.max(log_p_prev, axis=1)
    # previous
    p_prev = T.exp(log_p_prev - k)
    p_prev = T.set_subtensor(p_prev[:,t:],0)

    _result = p_prev
    # add shift of previous
    _result = T.inc_subtensor(_result[1:],   p_prev[:-1])
    # add skips of previous
    _result = T.inc_subtensor(_result[3::2], p_prev[1:-2:2])

    # current
    log_result = T.switch(t > 0, T.log(_result) + k, T.zeros_like(_result))
    t = t + 2
    _recurrence = log_result + log_p_curr
    recurrence = T.set_subtensor(_recurrence[t:], np.float32(0))
    non_inf = idx < t

    return t, recurrence, non_inf


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
        grads = T.grad(-cost[-1,-1],wrt=[baseline.W_test])
        f = theano.function(inputs=[X],outputs=[cost,grads[0],log_forward[2]])
        return f(inputs)

    inputs = np.random.randn(10,5).astype(np.float32)
    orig = base(inputs)
    log_space, grads, non_inf = log_space(inputs)
    print log_space - orig
    print non_inf
    print grads

