import numpy as np

user_embed = np.load('./pretrain/yelp2018/mf.npz')

print(len(user_embed['user_embed']))
