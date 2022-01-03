import argparse

parser=argparse.ArgumentParser(description='UpsideDown setting')

parser.add_argument('--lr',type=float,default=1e-3)											### learning rate
parser.add_argument('--hidden_units',type=int,default=64)									### Hidden units
parser.add_argument('--command_factor',type=float,default=1e-2)								### Command factor
parser.add_argument('--memory_thersold',type=int,default=600)								### Memory thersold (how many eposide)
parser.add_argument('--batch_size',type=int,default=1024)									### Batch Size
# parser.add_argument('--last_few',type=int,default=100)										### Num  eposide in  end of buffer 
parser.add_argument('--generate_per_single_training',type=int,default=100)					### How many step for generate new eposide for training
parser.add_argument('--n_update_eposide',type=int,default=200)								#### How many step used for train
parser.add_argument('--eval_step',type=float,default=20)									### Use how many step used in evalute
parser.add_argument('--eval_step_every_k_step',type=int,default=5)							### Every K step to evalute
parser.add_argument('--seed',type=int,default=123)											### Seed
parser.add_argument('--save',type = str, default='')
parser.add_argument('--Path',type=str,default=r'C:\Users\USER\Desktop\Upside down\new folder\result\memory_thersold/')


