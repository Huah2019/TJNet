import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from net.TJNet import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
import configparser
from torch.nn.parallel import DataParallel


def structure_loss(pred, mask):
    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                  stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) *
                    valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p))
                    * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()

            lateral_map_3, lateral_map_2, lateral_map_1, edge_map, coarse_map = model(
                images)

            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            losse = dice_loss(edge_map, edges)
            lossc = structure_loss(coarse_map, gts)
            loss = loss3 + loss2 + loss1 + 4 * losse + lossc

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}], [coarse: {:,.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss3.data, loss2.data, loss1.data, losse.data, lossc.data))
                logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                             '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}], [coarse: {:,.4f}]'.
                             format(datetime.now(), epoch, opt.epoch, i, total_step,
                                    loss3.data, loss2.data, loss1.data, losse.data, lossc.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss1': loss1.data, 'Lossc': lossc.data,
                                       'Losse': losse.data, 'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(
                    images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(
                    gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)
                grid_image = make_grid(
                    edges[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Edge', grid_image, step)

                # TensorboardX-Outputs
                res = coarse_map[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_coarse', torch.tensor(
                    res), step, dataformats='HW')

                res = lateral_map_1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(
                    res), step, dataformats='HW')

                res = edge_map[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_edge', torch.tensor(
                    res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info(
            '[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 80 == 0:
            torch.save(model.state_dict(), save_path +
                       'TJNet_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path +
                   'TJNet_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer, val_dataset, val_idx):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt,  name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            # gt /= (gt.max() + 1e-8)
            gt /= 255
            image = image.cuda()

            _, _, res, e, c = model(image)

            res = F.upsample(res, size=gt.shape,
                             mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / \
                (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE_{}'.format(val_dataset), torch.tensor(mae), global_step=epoch)
        print('val_dataset:{}, Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(
            val_dataset, epoch, mae, best_mae[val_idx], best_epoch[val_idx]))
        if epoch == 1:
            best_mae[val_idx] = mae
            best_epoch[val_idx] = 1
        else:
            if mae < best_mae[val_idx]:
                best_mae[val_idx] = mae
                best_epoch[val_idx] = epoch
                torch.save(model.state_dict(), save_path +
                           'TJNet_best_{}.pth'.format(val_dataset))
                print('Save state_dict successfully! Best epoch:{}, val_dataset:{}.'.format(epoch, val_dataset))
        logging.info(
            '[Val Info]:val_dataset:{} Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(val_dataset, epoch, mae, best_epoch[val_idx], best_mae[val_idx]))


if __name__ == '__main__':
    import argparse

    config = configparser.ConfigParser()
    config.read('./config.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=config['Train'].getint('epoch'), help='epoch number')
    parser.add_argument('--lr', type=float, default=config['Train'].getfloat('lr'), help='learning rate')
    parser.add_argument('--batchsize', type=int, default=config['Train'].getint('batchsize'),
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=config['Comm'].getint('trainsize'),
                        help='training dataset size')
    parser.add_argument('--clip', type=float, default=config['Train'].getfloat('clip'),
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=config['Train'].getfloat('decay_rate'), help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=config['Train'].getint('decay_epoch'),
                        help='every n epochs decay learning rate')
    __load__ = config['Train']['load']
    if __load__.lower() == 'none':
        __load__ = None
    parser.add_argument('--load', type=str, default=__load__,
                        help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default=config['Train']['train_root'],
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default=config['Train']['val_root'],
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str, default=config['Train']['save_path'],
                        help='the path to save model and log')
    parser.add_argument('--net_channel', type=int, default=config['Comm'].getint('net_channel'))
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    cudnn.benchmark = True

    # gpu_count = torch.cuda.device_count()
    # print("GPU count", gpu_count)
    
    model = Network(opt.net_channel)
    model = model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader_camo = test_dataset(image_root=opt.val_root + 'CAMO/Imgs/',
                              gt_root=opt.val_root + 'CAMO/GT/',
                              testsize=opt.trainsize)
    val_loader_chameleon = test_dataset(image_root=opt.val_root + 'CHAMELEON/Imgs/',
                              gt_root=opt.val_root + 'CHAMELEON/GT/',
                              testsize=opt.trainsize)
    val_loader_cod10k = test_dataset(image_root=opt.val_root + 'COD10K/Imgs/',
                              gt_root=opt.val_root + 'COD10K/GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    
    best_mae = [1, 1, 1]
    best_epoch =[0, 0, 0]

    # learning rate schedule
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=20, eta_min=1e-5)
    print("Start train...")
    
    

    
    for epoch in range(1, opt.epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch,
                           opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[
                          0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))

        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader_camo, model, epoch, save_path, writer, "camo" , 0)
        val(val_loader_chameleon, model, epoch, save_path, writer, "chameleon", 1)
        val(val_loader_cod10k, model, epoch, save_path, writer, "cod10k", 2)
