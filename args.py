import argparse
import os
import json
from datautils import load_UCR, load_HAR, load_mat, load_SMD, load_SWAT, load_WADI, load_MSL, load_PSM, load_SMAP, \
    load_SYN, load_GECCO, load_Swan

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='exp/test')
parser.add_argument('--data_path', type=str,default='data')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='SMD')
# model args
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--wave_length', type=int, default=32)
parser.add_argument('--mask_ratio', type=float, default=0.3)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--anomaly_ratio',type = float ,default=100)
parser.add_argument('--topk', type=int, default=10)

parser.add_argument('--dropout', type=float, default=0.2)#0.2
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--enable_res_parameter', type=int, default=1)
# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epoch_pretrain', type=int, default=50)
parser.add_argument('--load_pretrained_model', type=int, default=1)

# parser.add_argument('--eval_per_steps', type=int, default=16)
# parser.add_argument('--alpha', type=float, default=5.0)
# parser.add_argument('--beta', type=float, default=1.0)
# parser.add_argument('--reg_layers', type=int, default=3)
# parser.add_argument('--momentum', type=float, default=0.99)
# parser.add_argument('--vocab_size', type=int, default=192)
# parser.add_argument('--UCR_folder', type=str, default='PhonemeSpectra')
# parser.add_argument('--lr_decay_rate', type=float, default=1.)
# parser.add_argument('--lr_decay_steps', type=int, default=100)
# parser.add_argument('--weight_decay', type=float, default=0.01)
# parser.add_argument('--num_epoch', type=int, default=10000)

args = parser.parse_args()
args.data_path = 'data/'+args.dataset+'/'
if args.data_path is None:
    if args.dataset == 'ucr':
        Train_data_all, Train_data, Test_data = load_UCR(folder=args.UCR_folder)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'har':
        Train_data_all, Train_data, Test_data = load_HAR()
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'mat':
        Train_data_all, Train_data, Test_data = load_mat()
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'SMD':
        Train_data_all, Train_data, Test_data = load_SMD()
        args.num_class = 2
    elif args.dataset == 'SWAT':
        Train_data_all, Train_data, Test_data = load_SWAT()
        args.num_class = 2
    elif args.dataset == 'WADI':
        Train_data_all, Train_data, Test_data = load_WADI()
        args.num_class = 2
    elif args.dataset == 'MSL':
        Train_data_all, Train_data, Test_data = load_MSL()
        args.num_class = 2
    elif args.dataset == 'SMAP':
        Train_data_all, Train_data, Test_data = load_SMAP()
        args.num_class = 2
    elif args.dataset == 'PSM':
        Train_data_all, Train_data, Test_data = load_PSM()
        args.num_class = 2
    elif args.dataset == 'SYN':
        Train_data_all, Train_data, Test_data = load_SYN()
        args.num_class = 2
else:
    if args.dataset == 'ucr':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_UCR(path, folder=args.UCR_folder)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'har':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_HAR(path)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'mat':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_mat(path)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'SMD':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_SMD(path)
        args.num_class = 2
    elif args.dataset == 'SWAT':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_SWAT(path)
        args.num_class = 2
    elif args.dataset == 'WADI':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_WADI(path)
        args.num_class = 2
    elif args.dataset == 'MSL':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_MSL(path)
        args.num_class = 2
    elif args.dataset == 'SMAP':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_SMAP(path)
        args.num_class = 2
    elif args.dataset == 'PSM':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_PSM(path)
        args.num_class = 2
    elif args.dataset == 'SYN':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_SYN(path)
        args.num_class = 2
    elif args.dataset == 'GECCO':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_GECCO(path)
        args.num_class = 2
    elif args.dataset == 'SWAN':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_Swan(path)
        args.num_class = 2

# args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
# args.lr_decay_steps = args.eval_per_steps
args.save_path  = args.save_path+'//'+args.dataset
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
