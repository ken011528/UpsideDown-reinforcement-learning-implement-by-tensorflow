import gym
from arg import *

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sortedcontainers import SortedDict



class RL():
	def __init__(self,
		sess,
		env,
		n_s,
		n_a,
		args
		):

		if sess is None:
			self.sess=tf.Session()
		else:
			self.sess=sess

		self.args=args
		self.env=env
		self.env.seed(self.args.seed)

		self.n_s=n_s
		self.n_a=n_a

		self.init=True				### Use to detect is in init state or not, if Yes use random action

		self.ite_count=0				### Num of iter 

		self.dict=SortedDict()

		self.release=10
		self.reward_=200   ### use to clip reward

		self.save_index=0			### Save index


		self.network_model()
		self.saver=tf.compat.v1.train.Saver()
		tf.compat.v1.random.set_random_seed(args.seed)



	def network_model(self):

		def init_weights(input_):
			x = 1/(np.sqrt(input_))
			return tf.compat.v1.random_uniform_initializer(-x,x,seed=self.args.seed)

		def behavior_build_network():
			w_init = tf.compat.v1.initializers.variance_scaling(scale=np.sqrt(2/(1+np.sqrt(5)**2)),distribution='uniform',seed=self.args.seed)

			l1=tf.layers.dense(
				inputs=self.input_,
				units=self.args.hidden_units,
				kernel_initializer=w_init,
				bias_initializer=init_weights(self.n_s),
				activation='sigmoid',
				name='l1',
				)
			c1=tf.layers.dense(
				inputs=tf.concat((self.d_r,self.d_h),1),
				units=self.args.hidden_units,
				kernel_initializer=w_init,
				bias_initializer=init_weights(self.args.hidden_units),
				activation='sigmoid',
				name='c_out'
				)

			out_1=tf.math.multiply(l1,c1)

			l2=tf.layers.dense(
				inputs=out_1,
				units=self.n_a,
				activation=None,
				kernel_initializer=w_init,
				bias_initializer=init_weights(self.args.hidden_units),
				name='l2',
				)

			# b=tf.layers.dense(
			# 	inputs=l2,
			# 	units=self.n_a,
			# 	kernel_initializer=w_init,
			# 	bias_initializer=init_weights(self.args.hidden_units),
			# 	activation=None,
			# 	name='out'
			# 	)
			b=l2
			return b

		### 										ALL input 											
		self.input_=tf.compat.v1.placeholder(tf.float32,[None,self.n_s],'input_')

		self.c_in=tf.compat.v1.placeholder(tf.float32,[None,2],'c_in')
		self.d_h=tf.compat.v1.placeholder(tf.float32,[None,1],'d_h')
		self.d_r=tf.compat.v1.placeholder(tf.float32,[None,1],'d_r')

		self.a=tf.compat.v1.placeholder(tf.int32,[None,],'action')

		with tf.compat.v1.variable_scope('behavior_function'):
			self.b=behavior_build_network()
			self.b_softmax=tf.nn.softmax(self.b)
			self.a_out = tf.squeeze(tf.random.categorical(logits=self.b,num_samples=1,seed=self.args.seed))

		with tf.compat.v1.variable_scope('loss'):
			self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.b,labels=self.a))

		with tf.compat.v1.variable_scope('train'):
			self.train_op=tf.compat.v1.train.AdamOptimizer(self.args.lr).minimize(self.loss)


	def eval(self,eval_ite):
		test_reward=[]
		test_step=[]
		for i in range(self.args.eval_step):
			_,_,r_list,total_reward=self.play()
			test_reward.append(total_reward)
			test_step.append(len(r_list))
		print('ite: {},   reward: {:.3f},'.format(eval_ite,np.mean(test_reward)))
		return np.mean(test_reward)

	def train(self):
		self.feed()
		test=[]
		print('----------------using tensorflow with {} generate_step_per_single_training----------------'.format(self.args.generate_per_single_training))
		while True :
			self.learn()
			self.ite_count+=1
			self.feed()
			if (self.ite_count-1) % self.args.eval_step_every_k_step ==0:
				score=self.eval(self.ite_count-1)
				test.append(score)
				if len(test) % self.release == 0 or (self.ite_count-1)==0:
					self.saver.save(self.sess,r'C:\Users\USER\Desktop\Upside down\new folder\result\memory_thersold\tensorflow_model_{}_tensorflow_categorical_1.ckpt'.format(self.args.generate_per_single_training))
					np.save(r'C:\Users\USER\Desktop\Upside down\new folder\result\memory_thersold\tensorflow_reward_test_{}_{}__tensorflow_categorical_1.npy'.format(self.save_index,self.args.generate_per_single_training),test)
					print('saved')
					self.save_index+=1
					test=[]
					print((time.time()-start_)/60)
					start_ = time.time()

sess=tf.compat.v1.Session()
env=gym.make('LunarLander-v2')
n_s=env.observation_space.shape[0]
n_a=env.action_space.n
args=parser.parse_args()

RL_agent=RL(sess=sess,
	env=env,
	n_s=n_s,
	n_a=n_a,
	args=args)

RL_agent.train()

