import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

eps = 1e-40

def update_log_p(base,zeros,i,log_p_curr,log_p_prev):
    useful = T.min([(i + 1) * 2,log_p_prev.shape[0]])
    prev_useful = T.min([i * 2,log_p_prev.shape[0]])
    comm_factor = log_p_prev[0]
    skip_idxs = T.arange((base.shape[0] - 2)//2) * 2 + 1

    common_factor = log_p_prev[0]
    p_prev   = T.exp(log_p_prev[:prev_useful] - common_factor) 
    p_prev   = T.set_subtensor(base[:prev_useful],p_prev)
    _p_prev  = T.inc_subtensor(p_prev[1:],p_prev[:-1])
    __p_prev = T.inc_subtensor(_p_prev[skip_idxs + 2],p_prev[skip_idxs])
    updated_log_p_prev = T.log(__p_prev[:useful]) + common_factor

    log_p_next = T.inc_subtensor(
            zeros[:useful],
            log_p_curr[:useful] + updated_log_p_prev
        )
    return log_p_next


def path_probs(predict, Y):
    L = T.log(predict[:, Y])
    zeros = T.zeros_like(L[0])
    base = T.set_subtensor(zeros[:1],1)
    log_first = zeros
    def step(i,log_f_curr,log_b_curr,log_f_prev,log_b_prev):
        return update_log_p(base,zeros,i,log_f_curr,log_f_prev),\
                update_log_p(base,zeros,i,log_b_curr,log_b_prev)
    
    [log_f_probs,log_b_probs], _ = theano.scan(
            step,
            sequences=[T.arange(L.shape[0]),L,L[::-1, ::-1]],
            outputs_info=[log_first,log_first]
        )

    log_probs = log_f_probs + log_b_probs[::-1, ::-1] - L
    
    return log_probs

def cost(predict, Y):
    log_probs = path_probs(predict, Y)
    comm_factor = T.min(log_probs)
    idxs = 2 * T.arange(predict.shape[0])[::-1].dimshuffle(0,'x') + T.arange(Y.shape[0])
    relevant_probs = (idxs < predict.shape[0] * Y.shape[0]) * (idxs >= predict.shape[0])
    total_log_prob = T.log(T.sum(T.exp(log_probs - comm_factor) * relevant_probs)) + comm_factor
    return -total_log_prob


if __name__ == "__main__":
    import ctc
    probs = T.nnet.softmax(np.random.randn(20,11).astype(np.float32))
    labels = theano.shared(np.arange(11,dtype=np.int32))

    print ctc.cost(probs,labels).eval()
    print cost(probs,labels).eval()



