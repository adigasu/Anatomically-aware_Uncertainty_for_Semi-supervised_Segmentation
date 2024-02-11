import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNetDAE
from dataloaders import utils
from utils import ramps, losses
from dataloaders.medical_dataloader import AbdomenFLARE, RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from dataloaders.medical_dataloader import BatchSampler, RandomRescale, RandomCorrupt


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/FLARE_2021/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='DAE_L10', help='model_name')
parser.add_argument('--seed', type=int,  default=20221, help='random seed')
parser.add_argument('--max_iterations', type=int,  default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.1, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--num_classes', type=int,  default=5, help='Number of classes')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--nb_labels', type=int,  default=26, help='No. of labeled samples')
parser.add_argument('--total_labels', type=int,  default=260, help='Total no. of labeled samples')
parser.add_argument('--emb_dim', type=int,  default=512, help='embedding dimension of DAE')
parser.add_argument('--is_LS_noise', type=bool,  default=False, help='enable or disable LS (label space) noise in DAE')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../DAE_models/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = args.num_classes
input_size = (144, 144, 96)
is_LS_noise = args.is_LS_noise
emb_dim = args.emb_dim


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model():
        # Network definition
        net = VNetDAE(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False, is_LS_noise=is_LS_noise, input_size=input_size, emb_dim=emb_dim)
        model = net.cuda()
        return model

    model = create_model()

    db_train = AbdomenFLARE(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRescale(),
                          RandomRotFlip(),
                          RandomCrop(input_size),
                          RandomCorrupt(p=0.2),
                          ToTensor(),
                          ]))
    db_val = AbdomenFLARE(base_dir=train_data_path,
                       split='val',
                       transform = transforms.Compose([
                           CenterCrop(input_size),
                           ToTensor(),
                       ]))

    labeled_idxs = list(range(args.nb_labels))
    batch_sampler = BatchSampler(labeled_idxs, batch_size)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=4, shuffle=False,  num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    print("valloader ", len(valloader))

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations + 2000)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            noisy_batch = sampled_batch['noisy_label']
            noisy_batch = noisy_batch[:, None, :, :, :].type(torch.FloatTensor).cuda()

            noise = torch.clamp(torch.randn_like(noisy_batch) * 0.1, -0.2, 0.2)
            noisy_batch += noise

            outputs, emb = model(noisy_batch)  # 4*64* 3, 256
            outputs = torch.sigmoid(outputs)

            label_batch_oh = torch.zeros_like(outputs)
            for i in range(num_classes):
                label_batch_oh[:, i, :, :, :] = (label_batch == i)
            label_batch_oh = label_batch_oh.type(torch.FloatTensor).cuda()

            # Reconstruction loss
            loss_rec = F.mse_loss(outputs, label_batch_oh)
            dsc_score_all, loss_dice = losses.dice_loss_all(outputs, label_batch)
            loss = 0.5 * (loss_rec + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num = iter_num + 1
            lr_ = scheduler.get_last_lr()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_rec', loss_rec, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('loss/mean_label', label_batch.type(torch.FloatTensor).mean(), iter_num)
            writer.add_scalar('loss/mean_pred', outputs.mean(), iter_num)

            logging.info('iteration %d : loss: %f, loss_dice: %f' %
                         (iter_num, loss.item(), loss_dice.item()))

            if iter_num % 50 == 0:
                image = volume_batch[0, :1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/image', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Label', grid_image, iter_num)
                
                image = noisy_batch[0, 0, :, :, 20:61:10].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/noisy_Label', grid_image, iter_num)

                image = torch.max(outputs[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)


            # Best model
            if iter_num > 1000 and iter_num % len(trainloader) == 0:
                model.eval()
                
                val_loss_all = 0.0
                val_dice_all = 0.0
                for val_i, val_batch in enumerate(valloader):
                    val_img, val_gt = val_batch['image'], val_batch['label']
                    val_gt_ip = val_gt[:, None, :, :, :].type(torch.FloatTensor).cuda()
                    val_img, val_gt = val_img.cuda(), val_gt.cuda()
                    val_out, _ = model(val_gt_ip)

                    val_loss_seg = F.cross_entropy(val_out, val_gt)
                    val_out_soft = torch.sigmoid(val_out)
                    val_dice_score, val_dice_loss = losses.dice_loss_all(val_out_soft, val_gt)
                    val_loss = 0.5*(val_loss_seg + val_dice_loss)
                    val_loss_all += val_loss.cpu().data.numpy()
                    val_dice_all += val_dice_score.cpu().data.numpy()

                val_loss_all = val_loss_all / len(valloader)
                val_dice_all = val_dice_all / len(valloader)

                score = val_dice_all[1:].mean()
                logging.info('[val] iteration %d : loss: %f, dice:  %f' % (iter_num, val_loss_all, score * 100))

                if score > best_performance:
                    best_performance = score
                    save_mode_path = os.path.join(snapshot_path, 'best_model_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                writer.add_scalar('val/avg_abdomen_bkg', val_dice_all[0], iter_num)
                writer.add_scalar('val/avg_abdomen_liver', val_dice_all[1], iter_num)
                writer.add_scalar('val/avg_abdomen_kidney', val_dice_all[2], iter_num)
                writer.add_scalar('val/avg_abdomen_spleen', val_dice_all[3], iter_num)
                writer.add_scalar('val/avg_abdomen_pancreas', val_dice_all[4], iter_num)
                model.train()

            ## Save model
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
