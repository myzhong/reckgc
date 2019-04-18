import sys
import numpy as np 
import collections
import pickle
import numpy as np 
def new_kg():
	fr = open('./data/kg.dat')
	data = fr.readlines()
	fr.close()
	fr = open('./data/new_kg.dat','w')
	for line in data:
		relation = line.split('::')[1]
		if relation != 'rated' and relation != 'title':
		# if line.split('::')[1] != 'rated' :
			fr.write(line)

def read_ratings():
	fr = open('./data/ratings.dat')
	user_dict = {}
	item_dict = {}
	data = fr.readlines()
	fr.close()
	for line in data:
		line = line.strip()
		listfromline = line.split('::')
		if int(listfromline[2])>3:
			u = 'u'+listfromline[0]
			i = 'm'+listfromline[1]
			if u not in user_dict:
				user_dict[u] = [i]
			else:
				user_dict[u].append(i)
			if i not in item_dict:
				item_dict[i] = [u]
			else:
				item_dict[i].append(u)
	min_user = 5
	min_item = 2
	flag = True
	while flag:
		key = list(user_dict.keys())
		flag = False
		for user in key:
			if len(user_dict[user])<min_user:

				for item in user_dict[user]:
					item_dict[item].remove(user)
				del user_dict[user]
				flag = True
		for item in list(item_dict.keys()):
			if len(item_dict[item])<min_item:
				for user in item_dict[item]:
					user_dict[user].remove(item)
				del item_dict[item]

	mean_user = np.mean([len(user_dict[key]) for key in user_dict.keys()])
	mean_item = np.mean([len(item_dict[key]) for key in item_dict.keys()])
	num_rating = np.sum([len(user_dict[key]) for key in user_dict])
	print('mean_user: ', mean_user)
	print('mean_item: ', mean_item)
	print('num_user: ', len(user_dict))
	print('num_item: ', len(item_dict))
	print('num_rating: ', num_rating)
	return user_dict, item_dict

def read_kg():
	fr = open('./data/new_kg.dat')
	data = fr.readlines()
	fr.close()
	hrt = []
	entity = {}
	relation = {}
	for line in data:
		line = line.strip()
		listfromline = line.split('::')
		h,r,t = listfromline[:3]
		if t.isdigit():
			t = r[0]+t
		hrt.append((h,r,t))
		entity[h] = entity.get(h,0)+1
		entity[t] = entity.get(t,0)+1
		relation[r] = relation.get(r,0)+1
	print (relation)
	return hrt, entity, relation

def get_data(user_dict, item_dict, hrt, entity, relation):
	for user in user_dict:
		if user not in  entity:
			print(user)
	for item in item_dict:
		if item not in entity:
			print(item)

	# entity2idx = dict([(key, i) for i,key in enumerate(entity)])
	entity2idx = {}

	idx = 0
	for user in user_dict:
		entity2idx[user] = idx
		idx+=1
	for item in item_dict:
		entity2idx[item] = idx
		idx+=1

	for e in entity:
		if e not in entity2idx:
			entity2idx[e] = idx
			idx+=1









	relation2idx = dict([(key, i) for i,key in enumerate(relation)])
	num_user = len(user_dict)
	num_item = len(item_dict)
	num_entity = len(entity)
	num_relation = len(relation)
	print('num_entity, num_relation: ', num_entity, num_relation)

	user_entity = {}
	item_entity = {}

	tv_user = [[],[],[]]
	tv_item = [[],[],[]]

	for user in user_dict:
		user_idx = entity2idx[user]
		for item in user_dict[user]:
			item_idx = entity2idx[item]
			rand = np.random.random()
			if rand<0.7:
				tv_user[0].append(user_idx)
				tv_item[0].append(item_idx)
			elif rand<0.8:
				tv_user[1].append(user_idx)
				tv_item[1].append(item_idx)
			else:
				tv_user[2].append(user_idx)
				tv_item[2].append(item_idx)

	tv_h = [[],[],[]]
	tv_r = [[],[],[]]
	tv_t = [[],[],[]]
	# user_entity = np.zeros((num_entity, num_entity))
	hrt_mat = np.zeros((num_entity, num_entity))
	# associate_entity = {}

	for h,r,t in hrt:
		h_idx = entity2idx[h]
		r_idx = relation2idx[r]
		t_idx = entity2idx[t]
		# if h in user_dict:
		hrt_mat[h_idx, t_idx] = 1
		# if h_idx not in associate_entity:
		# 	associate_entity[h_idx] = [t_idx]
		# else:
		# 	associate_entity[h_idx].append(t_idx)
	
		rand = np.random.random()
		if rand<0.7:
			tv_h[0].append(h_idx)
			tv_r[0].append(r_idx)
			tv_t[0].append(t_idx)
		elif rand<0.8:
			tv_h[1].append(h_idx)
			tv_r[1].append(r_idx)
			tv_t[1].append(t_idx)
		else:
			tv_h[2].append(h_idx)
			tv_r[2].append(r_idx)
			tv_t[2].append(t_idx)

	associate_entity = []
	for i,line in enumerate(hrt_mat):
		entity_idx = np.argsort(line)[::-1][:4]
		# entity_ass = np.sort(line)[::-1][:5]
		associate_entity.append(entity_idx)
		# print (entity_idx, entity_ass)



	data = {}
	data['tv_user'] = tv_user
	data['tv_item'] = tv_item
	data['tv_h'] = tv_h
	data['tv_r'] = tv_r
	data['tv_t'] = tv_t
	data['num_user'] = num_user
	data['num_item'] = num_item
	data['num_entity'] = num_entity
	data['num_relation'] = num_relation
	data['associate_entity'] = associate_entity
	# data['associate_entity'] = associate_entity
	fr = open('data.pkl','wb')
	pickle.dump(data,fr)
	fr.close()




def get_train_s_neighbor(n_neighbor = 20):
	# n_neighbor = int(sys.argv[1])
	fr = open('data.pkl','rb')
	data = pickle.load(fr)
	fr.close()
	train_user = data['tv_user'][0]
	train_item = data['tv_item'][0]

	num_user = data['num_user']
	num_item = data['num_item']
	num_entity = data['num_entity']



	# tv_user = data['val_user']+data['test_user']
	# tv_item = data['val_item']+data['test_item']

	# bridge = np.zeros((num_user, num_item))
	# bridge = get_bridge(num_user, num_item, tv_user, tv_item)


	ratings = np.zeros((num_entity, num_entity))
	for u,i in zip(train_user,train_item):
		ratings[u,i] = 1


	ratings = ratings[:num_user]
	ratings = ratings[:,num_user:(num_user+num_item)]

	s_neighbor,_ = get_s_neighbor(ratings, n_neighbor)

	# np.save('train_s_mat',similar_mat)
	np.save('train_s_neighbor', s_neighbor[:num_user])
	# np.save('train_r_neighbor', r_neighbor)

def get_s_neighbor(ratings_mat, n_neighbor=20):
	num_user = len(ratings_mat)
	dot_mat = np.dot(ratings_mat, ratings_mat.T)*(1-np.eye(num_user))

	# dot_mat = dot_mat[:num_user]
	# dot_mat = dot_mat[:,num_user:(num_user+num_item)]

	sum_vec = np.sqrt(np.sum(ratings_mat**2, axis=1))

	sum_mat = np.dot(sum_vec.reshape((-1,1)), sum_vec.reshape((1,-1)))

	similar_mat = dot_mat/sum_mat
	# np.save('similar',similar_mat)
	s_neighbor = []
	s_neighbor_s = []
	# n_neighbor = 20
	for item in similar_mat:
		s_neighbor.append(np.argsort(item)[::-1][:n_neighbor])
		# s_neighbor_s.append(np.sort(item)[::-1][:n_neighbor])
	return np.array(s_neighbor), similar_mat




if __name__ == '__main__':
	# new_kg()


	# user_dict,item_dict = read_ratings()
	# hrt,entity,relation = read_kg()
	# get_data(user_dict, item_dict, hrt, entity, relation)


	get_train_s_neighbor()