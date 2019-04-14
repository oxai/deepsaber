import sys
sys.path.append("/users/guillefix/beatsaber/base")
sys.path.append("/users/guillefix/beatsaber")
#sys.path.append("/home/guillefix/code/beatsaber/base")
#sys.path.append("/home/guillefix/code/beatsaber")
import time
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
import random

#sys.argv.append("--data_dir=../DataE")
sys.argv.append("--data_dir=../AugData")
# sys.argv.append("--dataset_name=mfcc")
#sys.argv.append("--dataset_name=mfcc_look_ahead")
sys.argv.append("--dataset_name=mfcc_reduced_states_look_ahead")
#sys.argv.append("--model=adv_wavenet")
sys.argv.append("--model=wavenet")
#sys.argv.append("--frequency_gen_updates=2")
sys.argv.append("--batch_size=1")
#sys.argv.append("--output_length=767") # needs to be at least the receptive field (in time points) + 1 if using the GAN (adv_wavenet model)!
                                      # this is because the song part of the input will have length 94+94 for an output of length 95; then we can take the second part of the song input with the first 94 outputs as input to discriminator
sys.argv.append("--output_length=64") 
sys.argv.append("--num_windows=10")
sys.argv.append("--gpu_ids=0")
sys.argv.append("--nepoch=500")
sys.argv.append("--nepoch_decay=500")
sys.argv.append("--layers=7")
sys.argv.append("--blocks=3")
#sys.argv.append("--layers=5")
#sys.argv.append("--blocks=3")
sys.argv.append("--dilation_channels=512")
sys.argv.append("--residual_channels=256")
sys.argv.append("--skip_channels=256")
sys.argv.append("--end_channels=512")
sys.argv.append("--print_freq=10")
#sys.argv.append("--print_freq=1")
#sys.argv.append("--experiment_name=reduced_states_lookahead_likelihood")
#sys.argv.append("--experiment_name=reduced_states_lookahead_biggan")
#sys.argv.append("--experiment_name=reduced_states_gan_exp")
#sys.argv.append("--experiment_name=reduced_states_gan_exp_smoothedinput")
#sys.argv.append("--experiment_name=biggan_exp_smoothedinput")
sys.argv.append("--experiment_name=zeropad_entropy_regularization")
sys.argv.append("--save_by_iter")
sys.argv.append("--save_latest_freq=1000")
sys.argv.append("--workers=0")
#sys.argv.append("--dropout_p=0.5")
#sys.argv.append("--loss_ce_weight=0.0")
sys.argv.append("--entropy_loss_coeff=0.1")
#sys.argv.append("--load")
#sys.argv.append("--load_iter=2000")

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup()
    if not opt.gpu_ids:
        receptive_field = model.net.receptive_field
    else:
        receptive_field = model.net.module.receptive_field

    print("Receptive field is "+str(receptive_field)+" time points")
    print("Receptive field is "+str(receptive_field/opt.beat_subdivision)+" beats")
    train_dataset = create_dataset(opt, receptive_field=receptive_field)
    train_dataset.setup()
    train_dataloader = create_dataloader(train_dataset)
    if opt.val_epoch_freq:
        val_dataset = create_dataset(opt, validation_phase=True, receptive_field=receptive_field)
        val_dataset.setup()
        val_dataloader = create_dataloader(val_dataset)
    print('#training songs = {:d}'.format(len(train_dataset)))

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
            if opt.model == "adv_wavenet" and random.randint(1,opt.frequency_gen_updates) == opt.frequency_gen_updates:
                #I'm doing this probabilistically because I'm not sure if train_dataloader reshuffles the data between epochs
                #print("Optimizing generator")
                model.optimize_parameters(optimize_generator=True)
            else:
                #print("Optimizing discriminator")
                model.optimize_parameters()
            if total_steps % opt.display_freq == 0 or total_steps % opt.print_freq == 0:
                model.evaluate_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(losses)
                metrics = model.get_current_metrics()
                print(metrics)
                t = (time.time() - iter_start_time) / opt.batch_size

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

        if opt.val_epoch_freq and epoch % opt.val_epoch_freq == 0:
            val_start_time = time.time()
            with model.start_validation() as update_validation_meters:
                if opt.eval:
                    model.eval()
                for j, data in enumerate(val_dataloader):
                    val_start_time = time.time()
                    model.set_input(data)
                    model.test()
                    model.evaluate_parameters()
                    update_validation_meters()
            losses_val = model.get_current_losses(is_val=True)
            metrics_val = model.get_current_metrics(is_val=True)
            print("Validated parameters at epoch {:d} \t Time Taken: {:d} sec".format(epoch, int(time.time() - val_start_time)))
