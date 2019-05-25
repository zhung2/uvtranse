import time
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from options.train_options import TrainOptions
from models.models import Model
from datasets.vrd import VrdDataset


def train():
    opt = TrainOptions().parse()
    train_dataset = VrdDataset(opt.dataroot, split='train', net=opt.feat_net, use_lang=opt.use_lang)
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                   shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))
    val_dataset = VrdDataset(opt.dataroot, split='val', net=opt.feat_net, use_lang=opt.use_lang)
    val_data_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                                 shuffle=opt.serial_batches, num_workers=int(opt.nThreads))

    model = Model(opt)

    total_steps = 0
    batch = 0
    n_train_batches = len(train_data_loader)
    for epoch in range(opt.epoch_count, opt.niter + opt.epoch_count):
        loss_temp = 0
        epoch_start_time = time.time()
        epoch_iter = 0
        for i_batch, data_dict in enumerate(train_data_loader):
            batch += 1
            if opt.loss == 'kl':
                alpha = model.update_alpha(batch*1./n_train_batches)

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data_dict)
            model.optimize()
            loss = model.get_loss()
            loss_temp = loss_temp + loss * opt.batchSize

            # print statistics
            if epoch_iter % opt.print_epoch_iter_freq == 0:
                if epoch_iter > 0:
                    loss_temp = loss_temp / opt.print_epoch_iter_freq
                #print('epoch: {:d}, epoch_iter: {:d}, loss: {:.3f}'.format(epoch, epoch_iter, loss.cpu().data[0]))
                print('Epoch: {:d} \t Epoch_iter: {:d} \t Training Loss: {:.4f}'.format(epoch, epoch_iter, loss_temp))
                loss_temp = 0



        #if total_steps % opt.save_latest_freq == 0:
            #print('saving the latest model (epoch {:d}, total_steps {:d})'.format(epoch, total_steps))
            #model.save_model('latest')

        if epoch % opt.val_epoch_freq == 0:
            val_loss, val_true_loss = validate(model, val_data_loader, opt)
            print('=============== Epoch: {:d} \t Validation Loss: {:.4f} \t True Loss : {:.4f} ==============='.format(
                epoch, val_loss, val_true_loss))


        #lr = model.update_learning_rate(val_loss)
        lr = model.update_learning_rate(val_loss)
        if opt.loss == 'kl':
            print('[ End of epoch {:d} / {:d} \t Time Taken: {:f} sec \t Learning rate: {:.2e} \t alpha: {:.2e}]'.format(
                epoch, opt.niter + opt.epoch_count - 1, time.time() - epoch_start_time, lr, alpha))
        else:
            print('[ End of epoch {:d} / {:d} \t Time Taken: {:f} sec \t Learning rate: {:.2e}]'.format(
                epoch, opt.niter + opt.epoch_count - 1, time.time() - epoch_start_time, lr))

        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch {:d} \t iters {:d}'.format(epoch, total_steps))
            #model.save_model('latest')
            model.save_model(epoch)

def validate(model, val_data_loader, opt):
    total_loss = 0.0
    true_loss = 0.0
    for i, data_dict in enumerate(val_data_loader):
        if isinstance(data_dict['sub_loc'], int):
            continue
        model.set_input(data_dict)
        loss, true_val_loss = model.get_validate_loss()

        total_loss += loss
        true_loss += true_val_loss

    return total_loss / len(val_data_loader), true_val_loss / len(val_data_loader)

if __name__ == '__main__':
    train()
        

