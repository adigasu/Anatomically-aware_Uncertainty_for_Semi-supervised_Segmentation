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

from networks.vnet import VNet, VNetDAE
from dataloaders import utils
from utils import ramps, losses
from dataloaders.medical_dataloader import AbdomenFLARE, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/FLARE_2021/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='L10_r1', help='model_name')
parser.add_argument('--seed', type=int,  default=20221, help='random seed')
parser.add_argument('--max_iterations', type=int,  default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.1, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--num_classes', type=int,  default=5, help='Number of classes')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--nb_labels', type=int,  default=26, help='No. of labeled samples')
parser.add_argument('--total_labels', type=int,  default=260, help='Total no. of labeled samples')
parser.add_argument('--model_AE', type=str,  default='DAE_L10/best_model.pth', help='DAE model path')
parser.add_argument('--emb_dim', type=int,  default=512, help='emb_dim')
parser.add_argument('--is_LS_noise', type=bool,  default=True, help='enable or disable LS (label space) noise in DAE')
parser.add_argument('--gamma', type=float,  default=1.0, help='uncertainty weight')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../DAE_MT_models/" + args.exp + "/"
snapshot_path_DAE = "../DAE_models/" + args.model_DAE

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
patch_size = (144, 144, 96)
is_LS_noise = args.is_LS_noise
emb_dim = args.emb_dim


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


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


    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    def load_DAE_model(detach=True):
        save_mode_path = os.path.join(snapshot_path_DAE)
        model = VNetDAE(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False, input_size=patch_size, is_LS_noise=is_LS_noise, emb_dim=emb_dim).cuda()
        if detach:
            for param in model.parameters():
                param.detach_()
        model.load_state_dict(torch.load(save_mode_path))
        return model

    # create models
    model = create_model()
    ema_model = create_model(ema=True)
    dae_model = load_DAE_model()

    db_train = AbdomenFLARE(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_val = AbdomenFLARE(base_dir=train_data_path,
                       split='val',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor(),
                       ]))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    labeled_idxs = list(range(args.nb_labels))
    unlabeled_idxs = list(range(args.nb_labels, args.total_labels))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=4, shuffle=False,  num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    print("valloader ", len(valloader))


    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations + 2000)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    b_time = time.time()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise
            outputs = model(volume_batch)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)

            ## calculate the loss CE + Dice
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            dsc_score_all, loss_seg_dice = losses.dice_loss_all(outputs_soft[:labeled_bs], label_batch[:labeled_bs])

            # get output of teacher model
            ema_output_soft = F.softmax(ema_output, dim=1)
            outputs_argmaxed = torch.argmax(ema_output_soft, dim=1)
            outputs_argmaxed = outputs_argmaxed[:,None,:,:,:].type(torch.FloatTensor).cuda()

            # get DAE output 
            dae_model.eval()
            dae_outputs, emb = dae_model(outputs_argmaxed)
            preds = torch.sigmoid(dae_outputs)

            '''
            # Entropy
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, input_size)
            '''
            # L2
            uncertainty = (preds - ema_output_soft) **2
            # certainty region
            certainty = torch.exp(-1.0 * args.gamma * uncertainty) 

            # consistency loss
            consistency_weight = get_current_consistency_weight(iter_num//200)
            consistency_dist = consistency_criterion(outputs, ema_output) #(batch, num_classes, patch_size)
            mask = (certainty).float()
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * (consistency_dist)
            loss = 0.5*(loss_seg+loss_seg_dice) + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            lr_ = scheduler.get_last_lr()
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = torch.max(preds[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DAE_output', grid_image, iter_num)
            
                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                uncertainty_m = torch.mean(uncertainty, dim=1, keepdim=True)
                image = uncertainty_m[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                mask_m = torch.mean(mask, dim=1, keepdim=True)
                image = mask_m[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/mask', grid_image, iter_num)

                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            # Best model
            if (iter_num > 1000 and iter_num % len(trainloader) == 0):
                model.eval()
                
                val_loss_all = 0.0
                val_dice_all = 0.0
                for val_i, val_batch in enumerate(valloader):
                    val_img, val_gt = val_batch['image'], val_batch['label']
                    val_img, val_gt = val_img.cuda(), val_gt.cuda()
                    val_out = model(val_img)

                    val_loss_seg = F.cross_entropy(val_out, val_gt)
                    val_out_soft = F.softmax(val_out, dim=1)
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
    e_time = time.time()
    logging.info("Total training time: {}".format(e_time - b_time))
    writer.close()
