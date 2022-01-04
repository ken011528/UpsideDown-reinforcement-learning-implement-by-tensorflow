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


	def action_choice(self,s,c,dr,dh):
		s=np.asarray(s,dtype=np.float32).reshape((1,self.n_s))
		dr=np.asarray(dr).reshape((-1,1))
		dh=np.asarray(dh).reshape((-1,1))
		action = self.sess.run(self.a_out,{self.input_:s,self.d_r:dr,self.d_h:dh})
		return action


	def get_(self):
		if self.init:
			self.desire_r_init, self.desire_h_init=0,0
			return 
		h=[]
		r=[]

		for _ in range(self.args.generate_per_single_training):
			epoides=self.dict.popitem()
			h.append(len(epoides[1][0]))
			r.append(epoides[0])

		seleceted_eposide_len=np.mean(h)
		seleceted_eposide_mean=np.random.uniform(low=np.mean(r),high=(np.mean(r)+np.std(r)))
		self.desire_r_init,self.desire_h_init=seleceted_eposide_mean,seleceted_eposide_len

	def feed(self):
		self.get_()
		self.dict.clear()
		for _ in range(self.args.memory_thersold):
			state,action,reward,total_reward=self.play()
			self.dict.__setitem__(total_reward,(state,action,reward))
		self.init=False

	def play(self):
		s=self.env.reset()
		if self.ite_count==0:
			self.sess.run(tf.compat.v1.global_variables_initializer())

		state_list=[]
		action_list=[]
		reward_list=[]

		reward_total=0
		done=False

		desire_h=self.desire_h_init
		desire_r=self.desire_r_init

		while not done:
			c=np.asarray([desire_h,desire_r])

			if self.init:
				a=np.random.randint(self.n_a)
			else:
				a=self.action_choice(s,c,desire_r,desire_h)

			s_,r,done,_=self.env.step(a)

			state_list.append(s)
			action_list.append(a)
			reward_list.append(r)
			reward_total+=r

			desire_h=max(desire_h-1,1)
			desire_r=min(desire_r-r,self.reward_)

			s=s_

			if done:
					break
		return state_list,action_list,reward_list,reward_total


	def learn(self):
		if self.ite_count==0:
			self.sess.run(tf.compat.v1.global_variables_initializer())

		memory_dic=dict(self.dict)
		dic_value=list(memory_dic.values())

		for _ in range(self.args.n_update_eposide):
			state=[]
			dr=[]
			dh=[]
			true_a=[]
			c=[]
			indices=np.random.choice(len(dic_value),self.args.batch_size,replace=True) ######### random sample which eposide will use.
			tran=[dic_value[i] for i in indices]
			random_index=[np.random.choice(len(e[0])-2,1)  for e in tran]
			for idx_,tran_ in zip(random_index,tran):
				state.append(tran_[0][idx_[0]])
				dr.append(np.sum(tran_[2][idx_[0]:]))
				dh.append(len(tran_[0])-idx_[0])
				true_a.append(tran_[1][idx_[0]])
				c.append([np.sum(tran_[2][idx_[0]:]),len(tran_[0])-idx_[0]])


			command_ = np.asarray(c,dtype=np.float32).reshape(-1,2)
			s_t=np.asarray(state,dtype=np.float32)
			action=np.asarray([a_ for a_ in true_a])
			dr=np.asarray(dr,dtype=np.float32).reshape((-1,1))
			dh=np.asarray(dh,dtype=np.float32).reshape((-1,1))
			_,loss=self.sess.run([self.train_op,self.loss],{self.input_:s_t,self.c_in:command_,
				self.a:action,self.d_r:dr,self.d_h:dh})

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
# 					self.saver.save(self.sess,r'C:\Users\USER\Desktop\Upside down\new folder\result\memory_thersold\tensorflow_model_{}_tensorflow_categorical_1.ckpt'.format(self.args.generate_per_single_training))
					self.saver.save(self.sess,self.args.save_path+'\\'+'tensorflow_model_{}_tensorflow_categorical_1.ckpt'.format(self.args.generate_per_single_training))
# 					np.save(r'C:\Users\USER\Desktop\Upside down\new folder\result\memory_thersold\tensorflow_reward_test_{}_{}__tensorflow_categorical_1.npy'.format(self.save_index,self.args.generate_per_single_training),test)
					print('saved')
					self.save_index+=1
					test=[]
					print((time.time()-start_)/60)
					start_ = time.time()

# sess=tf.compat.v1.Session()
# # env=gym.make('LunarLander-v2')
# n_s=env.observation_space.shape[0]
# n_a=env.action_space.n
# args=parser.parse_args()

# RL_agent=RL(sess=sess,
# 	env=env,
# 	n_s=n_s,
# 	n_a=n_a,
# 	args=args)

# RL_agent.train()
