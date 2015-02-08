import theano
import theano.tensor as T
import numpy		 as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano.printing import Print

eps = 1e-40

def logplus(log_a,log_b):
	# Returns log (a + b)
	return log_a + T.log(1 + T.exp(log_b - log_a))

def log_(a):
	return T.log(T.clip(a,eps,1))

def exp_(a):
	return T.exp(T.clip(a,np.log(eps),30))

def path_probs(predict,Y):
	eye = T.eye(Y.shape[0])
	first = eye[0]
	mask0  = 1 - eye[0]
	mask1  = 1 - eye[1]

	alt_mask = T.cast(T.arange(Y.shape[0]) % 2,'float32')
	skip_mask = mask0 * mask1 * alt_mask
	prev_idx = T.arange(-1,Y.shape[0]-1)
	prev_prev_idx = T.arange(-2,Y.shape[0]-2)

	log_mask0     = log_(mask0)
	log_skip_mask = log_(skip_mask)
	log_first     = log_(first)

	def step(log_p_curr,log_p_prev):
		log_after_trans = logplus(
				log_p_prev,
				logplus(
					log_mask0     + log_p_prev[prev_idx],
					log_skip_mask + log_p_prev[prev_prev_idx]
				)
			)
		log_p_next = log_p_curr + log_after_trans
		return log_p_next

	L = T.log(predict[:,Y])

	log_f_probs,_ = theano.scan(step, sequences = [L],            outputs_info = [log_first])
	log_b_probs,_ = theano.scan(step, sequences = [L[::-1,::-1]], outputs_info = [log_first])

	log_probs = log_f_probs + log_b_probs[::-1,::-1]
	return log_probs,prev_idx,prev_prev_idx

def cost(predict,Y):
	log_probs,prev_idx,prev_prev_idx  = path_probs(predict,Y)
	max_log_prob = T.max(log_probs)

	# logsumexp.
	norm_probs = T.exp(log_probs - max_log_prob)
	norm_probs = Print("Norm_probs")(norm_probs)
	norm_total_log_prob = T.log(T.sum(norm_probs))
	
	log_total_prob = norm_total_log_prob + max_log_prob
	return -log_total_prob

