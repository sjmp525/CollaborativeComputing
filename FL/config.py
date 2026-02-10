import sys

# Network configration
SERVER_ADDR='192.168.215.128'
SERVER_PORT = 51000

K = 5# Number of devices
G = 3 # Number of groups

# Unique clients order
HOST2IP = {'client1':'192.168.215.132', 'client2':'192.168.215.133', 'client3':'192.168.215.135', 'client4': '192.168.215.136', 'client5': '192.168.215.137'}
# HOST2IP = {'client1':'192.168.215.132', 'client2':'192.168.215.133', 'client3':'192.168.215.135'}
CLIENTS_CONFIG= {'192.168.215.132':0, '192.168.215.133':1, '192.168.215.135':2, '192.168.215.136': 3, '192.168.215.137': 4}
#CLIENTS_CONFIG= {'192.168.215.132':0, '192.168.215.133':1, '192.168.215.135':2}
CLIENTS_LIST= [ '192.168.215.132', '192.168.215.133', '192.168.215.135', '192.168.215.136', '192.168.215.137']
#CLIENTS_LIST= [ '192.168.215.132', '192.168.215.133', '192.168.215.135']

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('FedAdapt')[0] + 'FedAdapt'
dataset_path = './dataset/'+ dataset_name +'/'
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5': [
		('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3),
		('M', 32, 32, 2, 32*16*16, 0),
		('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32),
		('M', 64, 64, 2, 64*8*8, 0),
		('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64),
		('D', 8*8*64, 128, 1, 64, 128*8*8*64),
		('D', 128, 10, 1, 10, 128*10)
	],
	'VGG8': [
		('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3),  # Conv1
		('C', 32, 32, 3, 32*32*32, 32*32*32*3*3*32), # Conv2
		('M', 32, 32, 2, 32*16*16, 0),              # MaxPool1
		('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), # Conv3
		('C', 64, 64, 3, 64*16*16, 64*16*16*3*3*64), # Conv4
		('M', 64, 64, 2, 64*8*8, 0),               # MaxPool2
		('C', 64, 128, 3, 128*8*8, 128*8*8*3*3*64), # Conv5
		('C', 128, 128, 3, 128*8*8, 128*8*8*3*3*128),# Conv6
		('M', 128, 128, 2, 128*4*4, 0),            # MaxPool3
		('D', 4*4*128, 128, 1, 128, 128*4*4*128),   # FC1
		('D', 128, 10, 1, 10, 128*10)               # FC2
	],
	'MobileNetV3': [
		('C', 3, 16, 3, 16 * 112 * 112, 16 * 112 * 112 * 3 * 3 * 3),
		('M', 16, 16, 2, 16 * 56 * 56, 0),  # Stage 1
		('C', 16, 16, 3, 16 * 56 * 56, 16 * 56 * 56 * 3 * 3 * 16),
		('C', 16, 24, 3, 24 * 56 * 56, 24 * 56 * 56 * 3 * 3 * 16),
		('M', 24, 24, 2, 24 * 28 * 28, 0),  # Stage 2
		('C', 24, 24, 3, 24 * 28 * 28, 24 * 28 * 28 * 3 * 3 * 24),
		('C', 24, 40, 3, 40 * 28 * 28, 40 * 28 * 28 * 3 * 3 * 24),
		('M', 40, 40, 2, 40 * 14 * 14, 0),  # Stage 3
		('C', 40, 40, 3, 40 * 14 * 14, 40 * 14 * 14 * 3 * 3 * 40),
		('C', 40, 80, 3, 80 * 14 * 14, 80 * 14 * 14 * 3 * 3 * 40),
		('M', 80, 80, 2, 80 * 7 * 7, 0),  # Stage 4
		('C', 80, 80, 3, 80 * 7 * 7, 80 * 7 * 7 * 3 * 3 * 80),
		('C', 80, 112, 3, 112 * 7 * 7, 112 * 7 * 7 * 3 * 3 * 80),
		# Stage 5
		('C', 112, 160, 3, 160 * 7 * 7, 160 * 7 * 7 * 3 * 3 * 112),
		('M', 160, 160, 2, 160 * 4 * 4, 0),  # Stage 6
		('C', 160, 160, 3, 160 * 4 * 4, 160 * 4 * 4 * 3 * 3 * 160),  # Stage 7
		('D', 160, 960, 1, 960, 960 * 160),  # FC
		('D', 960, 1280, 1, 1280, 1280 * 960),  # FC
		('D', 1280, 10, 1, 10, 1280 * 10)  # FC
	]

}
model_name = 'MobileNetV3'
# vgg5
# model_size = 1.28
# model_flops = 32.902
# total_flops = 8488192

# vgg8
# model_size = 1.587
# model_flops = 25.53
# total_flops = 25528832

# MobileNetV3
model_size = 5.51
model_flops = 213.96
total_flops = 55174400

# split_layer = [6,6,6,6,6] #Initial split layers
split_layer = [len(model_cfg[model_name]) - 1] * len(CLIENTS_LIST)
model_len = len(model_cfg[model_name])


# FL training configration
R = 100 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size


# RL training configration
max_episodes = 100         # max training episodes
max_timesteps = 100        # max timesteps in one episode
exploration_times = 20	   # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100				   # Batchsize
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'192.168.215.132': 5, '192.168.215.133': 5, '192.168.215.135': 5, '192.168.215.136': 5, '192.168.215.137': 5}  # infer times for each device

random = True
random_seed = 0
