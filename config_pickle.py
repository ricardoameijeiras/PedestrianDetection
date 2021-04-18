import pickle

with open('/home/benan/Detektion2/config.pickle', 'rb') as f_in:
	C = pickle.load(f_in)

with open('/home/benan/Detektion2/config.pickle', 'wb') as config_f:
	pickle.dump(C, config_f, protocol = 2)
