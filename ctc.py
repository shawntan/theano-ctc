import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

def interleave_blanks(Y):
    Y_ = T.alloc(-1,Y.shape[0] * 2 + 1)
    Y_ = T.set_subtensor(Y_[T.arange(Y.shape[0])*2 + 1],Y)
    return Y_

def create_skip_idxs(Y):
    skip_idxs = T.arange((Y.shape[0] - 3)//2) * 2 + 1
#    non_repeats = T.neq(Y[skip_idxs],Y[skip_idxs+2])
    return skip_idxs[non_repeats.nonzero()]


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    P.W_test = np.random.randn(5,5)
    def p_space(inputs):
        def recurrence(p_curr,t,p_prev):
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

        X = T.matrix('X')
        probs = T.nnet.softmax(T.dot(X,P.W_test))
        init_probs = T.alloc(np.float32(0),X.shape[1])
        init_probs = T.set_subtensor(init_probs[0],np.float32(1))
        forward, _ = theano.scan(
                fn=recurrence,
                sequences=[probs],
                outputs_info=[np.int32(0),init_probs]
            )

        forward[-1] = T.log(forward[-1])
        f = theano.function(inputs=[X],outputs=forward)
        idxs,vals = f(inputs)
        print idxs
        print vals

    def log_space(inputs):
        def recurrence(log_p_curr,t,log_p_prev):
            idxs = T.arange(log_p_prev.shape[0])

            k = T.max(log_p_prev)

            # previous
            p_prev = T.exp(log_p_prev - k)
            p_prev = T.set_subtensor(p_prev[t:],0)

            _result = p_prev
            # add shift of previous
            _result = T.inc_subtensor(_result[1:],   p_prev[:-1])
            # add skips of previous
            _result = T.inc_subtensor(_result[3::2], p_prev[1:-2:2])

            log_result = T.switch(t > 0, T.log(_result) + k, T.zeros_like(_result))
            # current
            _recurrence = log_result + log_p_curr
            t = t + 2
            recurrence = T.set_subtensor(_recurrence[t:], np.float32(0))
            return t, recurrence

        X = T.matrix('X')
        probs = T.nnet.softmax(T.dot(X,P.W_test))
        log_init_probs = T.alloc(np.float32(0), X.shape[1])
#        log_init_probs = T.set_subtensor(log_init_probs[0],np.float32(0))
        log_forward, _ = theano.scan(
                fn=recurrence,
                sequences=[T.log(probs)],
                outputs_info=[np.int32(0),log_init_probs]
            )
#        log_forward[-1] = T.exp(log_forward[-1])
        f = theano.function(inputs=[X],outputs=log_forward)
        idxs,vals = f(inputs)
        print idxs
        print vals
        print
#        print T.log(probs).eval({X:inputs})
        grads = T.grad(-log_forward[1][-1,-1],wrt=[P.W_test])
        print grads[0].eval({X:inputs})


    inputs =  np.random.randn(10,5).astype(np.float32)
    p_space(inputs)
    log_space(inputs)


def update_log_p(skip_idxs,zeros,active,log_p_curr,log_p_prev):
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
    # previous transitions.
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
    log_probs,mask = path_probs(predict, interleave_blanks(Y))
    common_factor = T.max(log_probs)
    total_log_prob = T.log(
        T.sum(T.exp(log_probs - common_factor)[mask.nonzero()])) + common_factor
    return -total_log_prob


