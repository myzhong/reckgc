import sys
import os
import numpy as np 
import tensorflow as tf 
import keras
from keras import backend as K
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
# from metrics import precision_k_curve,recall_k_curve,ndcg_k_curve
from utils import precision_k_curve, recall_k_curve, ndcg_k_curve,hr_k_curve, cos_sim, map_k_curve
from tensorflow.contrib import rnn
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  
from tensorflow.python.training.moving_averages import assign_moving_average


class Data_Loader():
	def __init__(self,batch_size):
		print("data loading...")
		fr = open('data.pkl','rb')

		self.data = pickle.load(fr)
		fr.close()
		self.num_entity = self.data['num_entity']
		self.num_item = self.data['num_item']
		self.num_user = self.data['num_user']
		self.num_relation = self.data['num_relation']

		self.tv_user = self.data['tv_user']
		self.tv_item = self.data['tv_item']

		self.tv_h = self.data['tv_h']
		self.tv_r = self.data['tv_r']
		self.tv_t = self.data['tv_t']

		self.batch_size = batch_size


		self.associate_entity = np.array(self.data['associate_entity'])

	def reset_data(self):
		print('\nresetting data...')
		self.train_user = self.tv_user[0]
		self.train_item = self.tv_item[0]
		# self.train_size = len(self.train_user)
		# pmtt = np.random.permutation(self.train_size)
		# self.train_user = np.array(self.train_user)[pmtt]
		# self.train_item = np.array(self.train_item)[pmtt]

		self.u_input = []
		self.i_input = []
		self.ui_label = []
		negative_sample_num = 4
		for u,i in zip(self.train_user, self.train_item):
			self.u_input += [u]*(negative_sample_num+1)
			negative_items = np.random.randint(self.num_user,self.num_user+self.num_item,negative_sample_num)
			self.i_input += [i]+list(negative_items)
			self.ui_label += [1]+[0]*negative_sample_num

		self.train_size = len(self.u_input)
		pmtt = np.random.permutation(self.train_size)
		self.u_input = np.array(self.u_input)[pmtt]
		self.i_input = np.array(self.i_input)[pmtt]
		self.ui_label = np.array(self.ui_label)[pmtt]



		negative_sample_num = 10
		self.h_input = []
		self.r_input = []
		self.t_input = []
		self.hrt_label = []
		for h,r,t in zip(self.tv_h[0], self.tv_r[0], self.tv_t[0]):
			self.h_input += [h]*(negative_sample_num+1)
			self.r_input += [r]*(negative_sample_num+1)
			negative_tails = np.random.randint(self.num_user+self.num_item, self.num_entity, negative_sample_num)
			self.t_input += [t]+list(negative_tails)
			self.hrt_label += [1]+[0]*negative_sample_num
		self.hrt_size = len(self.h_input)
		pmtt = np.random.permutation(self.hrt_size)
		self.h_input = np.array(self.h_input)[pmtt]
		self.r_input = np.array(self.r_input)[pmtt]
		self.t_input = np.array(self.t_input)[pmtt]
		self.hrt_label = np.array(self.hrt_label)[pmtt]





		self.ui_pointer = 0
		self.hrt_pointer = 0
		print(self.train_size)
		print(self.hrt_size)
		# print(len(self.tv_h[0]))

	def reset_pointer(self):
		self.ui_pointer = 0
		self.hrt_pointer = 0

	def next_batch(self):
		start = self.ui_pointer*self.batch_size
		end = (self.ui_pointer+1)*self.batch_size
		self.ui_pointer+=1

		if (self.ui_pointer+1)*self.batch_size>self.train_size:
			self.ui_pointer = 0


		hrt_start = self.hrt_pointer*self.batch_size
		hrt_end = (self.hrt_pointer+1)*self.batch_size
		self.hrt_pointer+=1
		if (self.hrt_pointer+1)*self.batch_size>self.hrt_size:
			self.hrt_pointer = 0



		return self.u_input[start:end], self.i_input[start:end], self.ui_label[start:end]\
				,self.h_input[hrt_start:hrt_end]\
				,self.r_input[hrt_start:hrt_end]\
				,self.t_input[hrt_start:hrt_end]\
				,self.hrt_label[hrt_start:hrt_end]



class Model():
	def __init__(self,batch_size, hidden_size, layers,
				num_entity, num_relation, associate_entity):

		print('model building...')
		self.user_input = tf.placeholder(tf.int32, shape=[None])
		self.item_input = tf.placeholder(tf.int32, shape=[None])

		self.ui_label = tf.placeholder(tf.float32, shape=[None,1])

		self.entity_embedding = tf.Variable(
			tf.random_uniform([num_entity, hidden_size], -1.0,1.0))

		self.relation_embedding = tf.Variable(
			tf.random_uniform([num_relation, hidden_size], -1.0,1.0))


		user_latent = tf.nn.embedding_lookup(self.entity_embedding, self.user_input)
		item_latent = tf.nn.embedding_lookup(self.entity_embedding, self.item_input)
		

		######################################### (user, item) pair 
		sim1 = tf.reduce_sum(tf.multiply(user_latent, item_latent), axis=1, keep_dims = True)
		vector1 = tf.concat([user_latent, item_latent,sim1, tf.multiply(user_latent, item_latent) ], axis=1)
		vector1 = tf.layers.batch_normalization(vector1)

		for i in range(len(layers)):
			hidden = Dense(layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v1_ui_hidden_' + str(i))
			vector1 = hidden(vector1)
		self.ui_logits = Dense(1, kernel_initializer='lecun_uniform', name = 'ui_pred')(vector1)

		self.ui_prediction = tf.nn.sigmoid(self.ui_logits)

		self.ui_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.ui_logits, labels = self.ui_label))


		##################################################loss function 
		reg_error = tf.nn.l2_loss(self.entity_embedding)\
					+tf.nn.l2_loss(self.relation_embedding)

		self.cost = self.ui_loss
		# self.cost += self.hrt_loss
		self.cost += 0.0001*reg_error
		# self.cost += 0.001*self.regularization(self.user_input, self.entity_embedding)

		self.train_op =  tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)




def sample_ui(u,i):
	user = []
	item = []
	for usr,itm in zip(u,i):
		rand = np.random.random()
		if rand<0.003:
			user.append(usr)
			item.append(itm)
	return user,item
def get_data(user,item,data_loader):
	data = data_loader.data 
	# c_item = range(data['num_item'])
	num_item = data['num_item']
	negative_items = 100
	u = [user]*negative_items
	i = [item]+np.random.randint(0,num_item,(negative_items-1)).tolist()
	ui_label = [1]+[0]*(negative_items-1)
	pmtt = np.random.permutation(negative_items)
	return np.array(u),\
			np.array(i)[pmtt],\
			np.array(ui_label)[pmtt]


def val(data_loader,sess, model, tv_user, tv_item):
	res_matrix = [[],[]]
	max_k=10
	metrics_num = 2
	f = [hr_k_curve,ndcg_k_curve]
	for user, item in zip(tv_user,tv_item):
		# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
		u,i, ui_label = get_data(user,item,data_loader)
		y_pred = sess.run([model.ui_prediction], feed_dict = {model.user_input:u,
													model.item_input:i,
													model.ui_label:ui_label.reshape((-1,1))})
		for i in range(metrics_num):
			res = f[i](ui_label.flatten(),y_pred[0].flatten(),max_k)
			res_matrix[i].append(res[:])

	res = np.mean(np.array(res_matrix), axis=1).T
	return res[-1,0]


def load_all_variable(sess):
		saver = tf.train.Saver(tf.global_variables())

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded all parameters success!!!')
		else:
			print (' [!] Loaded all parameters failed...')


def train(batch_size, data_loader, model):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		load_all_variable(sess)
		saver = tf.train.Saver(tf.global_variables())
		tv_user,tv_item = sample_ui(data_loader.data['tv_user'][2], data_loader.data['tv_item'][2])
		best_hr_10 = 0
		epochs_1 = 10
		epochs_2 = 50
		for i in range(epochs_1):
			data_loader.reset_data()
			total_batch = int(data_loader.train_size/batch_size)
			for e in range(epochs_2):
				data_loader.reset_pointer()
				for b in range(total_batch):
					iterations = i*epochs_2*total_batch+e*total_batch+b
					u_input, i_input, ui_label,\
					h_input,r_input,t_input,hrt_label = data_loader.next_batch()

					train_loss, _ = sess.run([model.cost, model.train_op], feed_dict={model.user_input: u_input,
																						model.item_input:i_input,
																						model.ui_label:ui_label.reshape((-1,1))})
					sys.stdout.write('\r {}/{} epoch, {}/{} batch, train loss:{}'.\
									format(i,e,b,total_batch,train_loss))


					if(iterations)%5000==0:
						hr_10 = val(data_loader, sess, model, tv_user, tv_item)
						# print('\n',hr_10)
						if hr_10>best_hr_10:
							print('\n', hr_10)
							best_hr_10 = hr_10
							saver.save(sess, checkpoint_dir+'model.ckpt', global_step = iterations)




def test(data_loader, model):
	with tf.Session() as sess:
		load_all_variable(sess)

		res_matrix = [[],[]]
		max_k=10
		metrics_num = 2
		f = [hr_k_curve,ndcg_k_curve]
		count = 0
		for user, item in zip(data_loader.data['tv_user'][2], data_loader.data['tv_item'][2]):
			# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
			u,i, ui_label = get_data(user,item,data_loader)

			y_pred = sess.run(model.ui_prediction, feed_dict = {model.user_input:u,
														model.item_input:i,
														model.ui_label:ui_label.reshape((-1,1))})


			# att = np.around(att,decimals = 4)[0]
			# for item in att:
			# fr.write(str(u[0])+':\t'+'\t'.join(map(str,att))+'\n')

			for i in range(metrics_num):
				res = f[i](ui_label.flatten(),y_pred.flatten(),max_k)
				res_matrix[i].append(res[:])


			count+=1
			if (count)%3000==0:
				print (np.mean(np.array(res_matrix),axis=1))
			sys.stdout.write("\ruser: "+str(count))
			sys.stdout.flush()
		print (np.mean(np.array(res_matrix),axis=1))
		
		res = np.mean(np.array(res_matrix), axis=1).T
		np.savetxt(checkpoint_dir+"hrt.dat", res, fmt = "%.5f", delimiter = '\t')


checkpoint_dir = './'+sys.argv[0].split('.')[0]+'/'
if __name__ == '__main__':
	batch_size = 256
	data_loader = Data_Loader(batch_size = batch_size)
	layers = eval('[64,16]')
	model = Model(batch_size = batch_size,
					hidden_size= 64,
					layers = layers,
					num_entity = data_loader.num_entity,
					num_relation = data_loader.num_relation,
					associate_entity = data_loader.associate_entity)


	if sys.argv[1] == 'train':
		train(batch_size, data_loader, model)
	else:
		print('rec testing...')
		test(data_loader, model)
		print('kgc testing...')
		# test_hrt(data_loader, model)
	# data_loader.reset_data()
	# data_loader.next_batch()