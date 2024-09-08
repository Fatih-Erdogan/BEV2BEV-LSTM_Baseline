import torch
import torch.optim as optim
import torch.nn as nn
import os
import random
from torch.utils.data import DataLoader
from dataloader.dataset import CollateFunction
import utils
import progressbar
import numpy as np
import json
from types import SimpleNamespace

from data_preprocess.preprocess_data import preprocess_bevs_and_waypoints, create_xy_bins


def main():
    with open("config.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)

    if opt.preprocess_data:
        inp = input("Do you really want to resize the images and process the waypoints?\n" +
                    "If already done, dont do it!\n (y to confirm): ")
        if inp == "y":
            print("Preprocess begins...")
            preprocess_bevs_and_waypoints(opt.data_root, (opt.image_size, opt.image_size), verbose=True)

    # if you want to visualize sth, this is not the file for it
    # delta_x_values, delta_y_values = get_delta_waypoints_histogram(opt.data_root, -1, show_plots=False, save_plots=False)

    # these are values obtained by manual inspection on the set
    min_delta, max_delta = -2, 2
    bin_lims_x, bin_lims_y = create_xy_bins(min_delta, max_delta, (opt.num_x_bins, opt.num_y_bins))
    collate_fn = CollateFunction(bin_lims_x, bin_lims_y)

    if opt.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % opt.model_dir)
        optimizer = opt.optimizer
        model_dir = opt.model_dir
        opt = saved_model['opt']
        opt.optimizer = optimizer
        opt.model_dir = model_dir
        opt.log_dir = '%s/continued' % opt.log_dir
    else:
        name = 'model=%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f' % \
               (opt.image_size, opt.image_size, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers,
                opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip,
                opt.beta)
        dataset = "road_only" if opt.channels == 1 else "with_objects"
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, dataset + "_" + opt.train_on, name)

    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

    # assert torch.cuda.is_available()  # comment out if needed
    dtype1 = torch.FloatTensor
    dtype2 = torch.IntTensor

    seed = opt.seed
    if torch.cuda.is_available():
        print("Random Seed: ", seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor

    else:
        print("CUDA WAS NOT AVAILABLE!")
        print("Didn't seed...")

    # ---------------- load the models  ----------------

    print(opt)

    # ---------------- optimizers ----------------
    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)

    import model.lstm as lstm_models

    if opt.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        prior = saved_model['prior']
    else:
        frame_predictor = lstm_models.lstm(opt.g_dim + opt.z_dim + (opt.a_dim * opt.num_actions_per_frame), opt.a_dim,
                                           opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers,
                                           opt.num_x_bins * opt.num_y_bins, opt.batch_size)
        posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers,
                                              opt.batch_size)
        prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
        frame_predictor.apply(utils.init_weights)
        posterior.apply(utils.init_weights)
        prior.apply(utils.init_weights)

    import model.vgg_128 as model

    if opt.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = model.encoder(opt.g_dim, opt.channels)
        decoder = model.decoder(opt.g_dim, opt.channels)
        encoder.apply(utils.init_weights)
        decoder.apply(utils.init_weights)

    frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # --------- loss functions ------------------------------------
    if opt.loss_fn == "mse":
        reconstruction_criterion = nn.MSELoss()
    elif opt.loss_fn == "l1":
        reconstruction_criterion = nn.L1Loss()
    else:
        raise NotImplementedError("Unknown loss function: %s" % opt.loss_fn)

    def kl_criterion(mu1, logvar1, mu2, logvar2):
        # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
        #   log( sqrt(
        #
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / opt.batch_size

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.is_available():
        frame_predictor.cuda()
        posterior.cuda()
        prior.cuda()
        encoder.cuda()
        decoder.cuda()

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)

    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_fn)

    def get_training_batch():
        while True:
            for sequence in train_loader:
                batch = utils.normalize_data(dtype1, dtype2, sequence)
                # returns x, actions
                yield batch

    training_batch_generator = get_training_batch()

    def get_testing_batch():
        while True:
            for sequence in test_loader:
                batch = utils.normalize_data(dtype1, dtype2, sequence)
                # returns x, actions
                yield batch

    testing_batch_generator = get_testing_batch()

    # --------- plotting funtions ------------------------------------
    def plot(x, epoch, actions=None):
        n_sample = 20
        gen_seq = [[] for _ in range(n_sample)]
        gt_seq = [x[i] for i in range(len(x))]

        for s in range(n_sample):
            frame_predictor.hidden = frame_predictor.init_hidden()
            posterior.hidden = posterior.init_hidden()
            prior.hidden = prior.init_hidden()
            gen_seq[s].append(x[0])
            x_in = x[0]
            for i in range(1, opt.n_eval):
                h = encoder(x_in)
                if opt.last_frame_skip or i < opt.n_past:
                    h, skip = h
                else:
                    h, _ = h
                if i < opt.n_past:
                    h_target = encoder(x[i])
                    h_target = h_target[0]
                    z_t, _, _ = posterior(h_target)
                    prior(h)

                    tiled_action = actions[i - 1]
                    frame_predictor(torch.cat([h, z_t], 1), tiled_action)

                    x_in = x[i]
                    gen_seq[s].append(x_in)
                else:
                    z_t, _, _ = prior(h)

                    tiled_action = actions[i - 1]
                    h = frame_predictor(torch.cat([h, z_t], 1), tiled_action)

                    x_in = decoder([h, skip])
                    gen_seq[s].append(x_in)

        to_plot = []
        gifs = [[] for _ in range(opt.n_eval)]
        n_row = min(opt.batch_size, 10)
        eval_mse = []
        for i in range(n_row):
            # ground truth sequence
            row = []
            for t in range(opt.n_eval):
                row.append(gt_seq[t][i])
            to_plot.append(row)

            # best sequence
            min_mse = 1e7
            for s in range(n_sample):
                mse = 0
                sample_eval_mse = 0
                for t in range(opt.n_eval):
                    mse += torch.sum((gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu()) ** 2)
                    sample_eval_mse += reconstruction_criterion(gt_seq[t][i].data.cpu(), gen_seq[s][t][i].data.cpu())
                if mse < min_mse:
                    min_mse = mse
                    min_idx = s
            eval_mse.append(sample_eval_mse / opt.n_eval)

            s_list = [min_idx]
            additional_samples = np.random.choice(n_sample, size=4, replace=False)
            additional_samples = additional_samples[additional_samples != min_idx]
            while len(additional_samples) < 4:
                new_samples = np.random.choice(n_sample, size=4 - len(additional_samples), replace=False)
                new_samples = new_samples[new_samples != min_idx]
                additional_samples = np.concatenate((additional_samples, new_samples))
            s_list.extend(additional_samples[:4])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row = []
                for t in range(opt.n_eval):
                    row.append(gen_seq[s][t][i])
                to_plot.append(row)

            for t in range(opt.n_eval):
                row = list()
                row.append(gt_seq[t][i])
                for ss in range(len(s_list)):
                    s = s_list[ss]
                    row.append(gen_seq[s][t][i])
                gifs[t].append(row)
        avg_eval_mse = np.mean(eval_mse)
        print({'eval/mse': avg_eval_mse})
        f_name = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
        utils.save_tensors_image(f_name, to_plot)
        f_name = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
        utils.save_gif(f_name, gifs)
        # add a return with loss !!!

    def eval(x, actions=None):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        mse = 0.0
        for i in range(1, opt.n_past + opt.n_future):
            h = encoder(x[i - 1])
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h = h[0]

            if i < opt.n_past:
                h_target = encoder(x[i])[0]
                z_t, _, _ = posterior(h_target)
                prior(h)
                tiled_action = actions[i - 1]
                frame_predictor(torch.cat([h, z_t], 1), tiled_action)
            else:
                z_t, _, _ = prior(h)
                tiled_action = actions[i - 1]
                h = frame_predictor(torch.cat([h, z_t], 1), tiled_action)

                x_pred = decoder([h, skip])
                mse += reconstruction_criterion(x_pred, x[i])

        return mse.data.cpu().numpy() / opt.n_future

    # --------- training functions ------------------------------------
    def train(x, actions=None):
        frame_predictor.zero_grad()
        posterior.zero_grad()
        prior.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()

        # initialize the hidden state.
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        mse = 0.0
        kld = 0.0

        for i in range(1, opt.n_past + opt.n_future):
            h = encoder(x[i - 1])
            h_target = encoder(x[i])[0]
            if opt.last_frame_skip or i < opt.n_past:
                # skip comes from the last ground truth frame that the model is able to look at (just before pred starts)
                h, skip = h
            else:
                h = h[0]
            # only used in training, the purpose of prior is to predict the distribution of the next frame;
            # during training, to have a stable training and make the frame predictor learn better,
            # the posterior is used and at the end the difference between the prior models' out is punished to approach
            # prior to the posterior
            z_t, mu, logvar = posterior(h_target)
            _, mu_p, logvar_p = prior(h)

            tiled_action = actions[i - 1]
            # use the current frames encoding, the action condition, and the prediction of the next frame's
            # distribution (which is given as posterior(z | x_i) but will be prior(z | x_(i-1)) ), try to come up with
            # h of the next frame
            h_pred = frame_predictor(torch.cat([h, z_t], 1), tiled_action)

            x_pred = decoder([h_pred, skip])

            mse += reconstruction_criterion(x_pred, x[i])
            kld += kl_criterion(mu, logvar, mu_p, logvar_p)

        loss = mse + kld * opt.beta
        loss.backward()     # takes soooo long :((

        frame_predictor_optimizer.step()
        posterior_optimizer.step()
        prior_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return mse.data.cpu().numpy() / (opt.n_past + opt.n_future - 1), kld.data.cpu().numpy() / (
                opt.n_future + opt.n_past - 1)

    # --------- training loop ------------------------------------
    train_minibatch_losses = list()
    eval_batch_losses = list()
    for epoch in range(opt.epochs):
        epoch_train_minibatch_losses = list()
        frame_predictor.train()
        posterior.train()
        prior.train()
        encoder.train()
        decoder.train()
        epoch_mse = 0
        epoch_kld = 0
        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
        for i in range(opt.epoch_size):
            progress.update(i + 1)
            x, actions = next(training_batch_generator)
            # train frame_predictor
            mse, kld = train(x, actions)  # log_gif=(i == opt.epoch_size - 1))
            epoch_mse += mse
            epoch_kld += kld
            epoch_train_minibatch_losses.append((mse, kld))
        train_minibatch_losses.append(epoch_train_minibatch_losses)

        progress.finish()
        utils.clear_progressbar()

        print('[%02d] training: mse loss: %.5f | kld loss: %.5f (%d)' % (
            epoch, (epoch_mse / opt.epoch_size), (epoch_kld / opt.epoch_size), (epoch + 1) * opt.epoch_size * opt.batch_size))

        # eval
        print("Starting evaluation...")
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()
        prior.eval()

        test_len = len(test_loader)
        progress = progressbar.ProgressBar(max_value=test_len).start()
        test_mse = 0.0
        for i in range(test_len):
            progress.update(i + 1)
            x, actions = next(testing_batch_generator)
            t_mse = eval(x, actions)
            test_mse += t_mse
        avg_test_mse = test_mse / test_len
        eval_batch_losses.append(avg_test_mse)
        print('[%02d] eval: mse loss: %.5f (%d)' % (epoch, avg_test_mse, (epoch + 1) * test_len * opt.batch_size))
        print("Ended evaluation!\n")

        print("Starting plotting...")
        # plot some stuff
        # sacrifice a batch :(
        x, actions = next(testing_batch_generator)
        with torch.no_grad():
            plot(x, epoch, actions=actions)
            # plot_rec(x, epoch, actions=actions)
        print("Ended plotting!\n")

        # save the losses
        loss_f_name = '%s/loss/out.json' % opt.log_dir
        utils.save_loss(loss_f_name, train_minibatch_losses, eval_batch_losses)

        # save the model
        torch.save({
            'encoder': encoder,
            'decoder': decoder,
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'prior': prior,
            'opt': opt},
            '%s/model.pth' % opt.log_dir)
        if epoch % 10 == 0:
            print('log dir: %s' % opt.log_dir)


if __name__ == '__main__':
    main()
