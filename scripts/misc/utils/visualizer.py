import os
import time
import visdom


class Visualizer:
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.is_train and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.experiment_name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        self.log_name = os.path.join(opt.checkpoints_dir, self.opt.experiment_name, 'loss_log.txt')
        self.image_size = 256
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        raise ConnectionError(
            "Could not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n")

    # losses: dictionary of error labels and values
    def plot_current_losses_metrics(self, epoch, epoch_progress, losses, metrics):
        """
        plot both metrics and losses
        :param epoch:
        :param epoch_progress:
        :param losses:
        :param metrics:
        :return:
        """
        if not hasattr(self, 'loss_data'):
            losses_legend = list(losses.keys()) + [loss + "_val" for loss in losses.keys() if
                                                   not loss.endswith('_val')]
            self.loss_data = {'X': [epoch - 1], 'Y': [[0] * len(losses_legend)], 'legend': losses_legend}
        self.loss_data['X'].append(epoch + epoch_progress)
        # fill with latest value if loss is not given for update
        self.loss_data['Y'].append([losses[k] if k in losses.keys() else self.loss_data['Y'][-1][i]
                                    for i, k in enumerate(self.loss_data['legend'])])
        if not hasattr(self, 'metric_data'):
            metrics_legend = list(metrics.keys()) + [metric + "_val" for metric in metrics.keys()]
            self.metric_data = {'X': [epoch - 1], 'Y': [[0] * len(metrics_legend)], 'legend': metrics_legend}
        self.metric_data['X'].append(epoch + epoch_progress)
        self.metric_data['Y'].append([metrics[k] if k in metrics.keys() else self.metric_data['Y'][-1][j]
                                      for j, k in enumerate(self.metric_data['legend'])])
        try:
            self.vis.line(
                X=np.stack([np.array(self.loss_data['X'])] * len(self.loss_data['legend']), 1),
                Y=np.array(self.loss_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
            self.vis.line(
                X=np.stack([np.array(self.metric_data['X'])] * len(self.metric_data['legend']), 1),
                Y=np.array(self.metric_data['Y']),
                opts={
                    'title': self.name + ' metric over time',
                    'legend': self.metric_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id + 1)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    def print_current_losses_metrics(self, epoch, i, losses, metrics, t, t_data):
        if i:  # iter is not given in validation (confusing?)
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        else:
            message = '(epoch: %d, validation) ' % (epoch)
        for k, v in (OrderedDict(losses, **metrics)).items():  # not displayed in correct order in python <3.6
            if not i:
                k = '_'.join(k.split('_')[0:-1])
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
