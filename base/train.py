import time
import os
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
from utils import utils


if __name__ == '__main__':
    opt = TrainOptions().parse()
    train_dataset = create_dataset(opt)
    train_dataloader = create_dataloader(train_dataset)

    model = create_model(opt)
    model.setup()
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.nepoch + opt.nepoch_decay):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.display_freq == 0 or total_steps % opt.print_freq == 0:
                model.evaluate_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                metrics = model.get_current_metrics()  # added by me
                t = (time.time() - iter_start_time) / opt.batch_size
                utils.print_current_losses_metrics(epoch, epoch_iter, losses, metrics, t, t_data)
                if opt.display_id > 0:
                    epoch_progress = float(epoch_iter) / (len(train_dataloader) * opt.batch_size)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

