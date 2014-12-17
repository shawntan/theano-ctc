import theano
import theano.tensor as T
import numpy		 as np
from theano_toolkit import utils as U
from theano_toolkit import updates

def path_probs(predict,Y):
	eye = T.eye(Y.shape[0])
	first = eye[0]
	mask0  = 1 - eye[0]
	mask1  = 1 - eye[1]
	alt_mask = T.cast(T.arange(Y.shape[0]) % 2,'float32')
	skip_mask = mask0 * mask1 * alt_mask
	prev_idx = T.arange(-1,Y.shape[0]-1)
	prev_prev_idx = T.arange(-2,Y.shape[0]-2)
	
	def step(p_curr,p_prev):
		p_next = p_curr *(
					p_prev +\
					mask0 * p_prev[prev_idx] +\
					skip_mask * p_prev[prev_prev_idx]
				)
		return p_next
	
	L = predict[:,Y]
	f_probs,_ = theano.scan(step,sequences = [L],outputs_info = [first])
	b_probs,_ = theano.scan(step,sequences = [L[::-1,::-1]],outputs_info = [first])
	
	probs = f_probs * b_probs[::-1,::-1] / predict[:,Y]
	
	return probs,prev_idx,prev_prev_idx

def cost(predict,Y):
	probs,prev_idx,prev_prev_idx  = path_probs(predict,Y)
	
	total_prob = T.sum(probs)
	return -T.log(total_prob)

