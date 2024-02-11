import os
import argparse
import torch
from networks.vnet import VNetDAE
from test_util import test_all_case


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/FLARE_2021/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='Vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--split', type=str,  default='test', help='train/val/test split')
parser.add_argument('--save', action='store_true',  default=False, help='save results')
parser.add_argument('--emb_dim', type=int,  default=512, help='emb_dim')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
model_path = '../DAE_models/'
snapshot_path = model_path + FLAGS.model
test_save_path = model_path + 'prediction/'+FLAGS.model+'_post/'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 5
patch_size = (144,144,96)
root_path = FLAGS.root_path
split = FLAGS.split
emb_dim = FLAGS.emb_dim


def test_calculate_metric():
    net = VNetVAE(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False, input_size=patch_size, emb_dim=emb_dim).cuda()
    save_mode_path = os.path.join(snapshot_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # Normalization : mean_percentile, full_volume_mean
    avg_metric = test_all_case(net, root_path, split, normalization='mean_percentile', num_classes=num_classes,
                               patch_size=patch_size, stride_xy=18, stride_z=4,
                               save_result=FLAGS.save, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
