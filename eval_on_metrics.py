import torch
import torch.nn as nn
from piq import ssim
import os
import random
from torch.utils.data import DataLoader
from dataloader.dataset import CollateFunction
import utils
import progressbar
import json
from types import SimpleNamespace

from data_preprocess.preprocess_data import create_xy_bins


def main():
    with open("config.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)

    # these are values obtained by manual inspection on the set
    min_delta, max_delta = -2, 2
    bin_lims_x, bin_lims_y = create_xy_bins(min_delta, max_delta, (opt.num_x_bins, opt.num_y_bins))
    collate_fn = CollateFunction(bin_lims_x, bin_lims_y)

    assert os.path.isdir(opt.model_dir), "For evaluation, a pretrained model directory is required!"

    # load model and evaluate the model at that checkpoint
    map_loc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    saved_model = torch.load('%s/model.pth' % opt.model_dir, map_location=map_loc)
    bs = opt.batch_size
    opt = saved_model['opt']
    opt.batch_size = bs

    # print(opt)
    # input("Do you want to continue? If not ctrl+c")

    os.makedirs('%s/evaluation/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/evaluation/plots/' % opt.log_dir, exist_ok=True)

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

    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.is_available():
        frame_predictor.cuda()
        posterior.cuda()
        prior.cuda()
        encoder.cuda()
        decoder.cuda()

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)

    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_fn)

    def get_testing_batch():
        while True:
            for sequence in test_loader:
                batch = utils.normalize_data(dtype1, dtype2, sequence)
                # returns x, actions
                yield batch

    # --------------------- Evaluation Metrics ---------------------
    # reconstruction criterion
    # mean intersection over union
    # structural similarity index measure

    testing_batch_generator = get_testing_batch()

    if opt.loss_fn == "mse":
        reconstruction_criterion = nn.MSELoss()
    elif opt.loss_fn == "l1":
        reconstruction_criterion = nn.L1Loss()
    else:
        raise NotImplementedError("Unknown loss function: %s" % opt.loss_fn)

    def calculate_mean_iou(pred, gt, standard=True):
        # for now, assume c is always 1
        """
        :param gt: tensor(bs * c * h * w) gt for n'th PREDICTION sequence, over all minibatch elements
        :param pred: tensor(bs * c * h * w) preds for n'th PREDICTION sequence, over all minibatch elements
        :return: float: mean intersection over union
        """
        if standard:
            scale = 1.0 / gt.max()
            gt = gt * scale
            pred = pred * scale

            threshold = 0.3
            gt = gt > threshold
            pred = pred > threshold
            intersection = torch.logical_and(pred, gt).float()
            union = torch.logical_or(pred, gt).float()

            intersection = intersection.sum(dim=[1, 2, 3])  # bs
            union = union.sum(dim=[1, 2, 3])  # bs
            iou = intersection / torch.clamp(union, min=1e-6)
            return iou.mean().item()

        else:
            union = torch.max(pred, gt)
            intersection = torch.min(pred, gt)

            c_inter = intersection.sum(dim=(1, 2, 3))  # bs
            c_uni = union.sum(dim=(1, 2, 3))  # bs
            m_iou = torch.mean(c_inter / c_uni)  # mean over batch for a seq element

            return m_iou.item()

    def calculate_ssim(pred, gt):
        # for now, assume c is always 1
        """
        :param gt: tensor(bs * c * h * w) gt for n'th PREDICTION sequence, over all minibatch elements
        :param pred: tensor(bs * c * h * w) preds for n'th PREDICTION sequence, over all minibatch elements
        :return: float: ssim
        """
        pred = pred.float()
        gt = gt.float()
        ssim_value = ssim(pred, gt, kernel_size=11, data_range=1.0, reduction="mean")

        return ssim_value.item()

    def eval(x, actions=None):
        # returns 3 lists of size (seq_len) containing mean minibatch evaluation results
        # the purpose is to visualize the increase in corruption as the sequence moves on
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        mse = list()
        miou = list()
        ssim = list()
        x_in = x[0].clone()
        for i in range(1, opt.n_past + opt.n_future):
            h = encoder(x_in)
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
                x_in = x[i].clone()
            else:
                z_t, _, _ = prior(h)
                tiled_action = actions[i - 1]
                h = frame_predictor(torch.cat([h, z_t], 1), tiled_action)

                x_pred = decoder([h, skip])
                x_in = x_pred.clone()

                mse.append(reconstruction_criterion(x_pred, x[i]))

                miou.append(calculate_mean_iou(x_pred.clamp(0, 1), x[i].clamp(0, 1), True))
                ssim.append(calculate_ssim(x_pred.clamp(0, 1), x[i].clamp(0, 1)))

        return torch.tensor(mse), torch.tensor(miou), torch.tensor(ssim)

    # --------- training loop ------------------------------------
    print("Starting evaluation...")
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()

    test_mse = torch.zeros(opt.n_future)
    test_m_iou = torch.zeros(opt.n_future)
    test_ssim = torch.zeros(opt.n_future)

    test_len = len(test_loader)
    progress = progressbar.ProgressBar(max_value=test_len).start()
    with torch.no_grad():
        for i in range(test_len):
            progress.update(i + 1)
            x, actions = next(testing_batch_generator)
            t_mse, t_m_iou, t_ssim = eval(x, actions)
            test_mse += t_mse
            test_m_iou += t_m_iou
            test_ssim += t_ssim
    avg_test_mse, avg_test_miou, avg_test_ssim = test_mse / test_len, test_m_iou / test_len, test_ssim / test_len
    avg_test_mse = avg_test_mse.numpy().tolist()
    avg_test_miou = avg_test_miou.numpy().tolist()
    avg_test_ssim = avg_test_ssim.numpy().tolist()
    print("MSE across sequence elements: ")
    print(avg_test_mse, end="\n\n")
    print("MIOU across sequence elements: ")
    print(avg_test_miou, end="\n\n")
    print("SSIM across sequence elements: ")
    print(avg_test_ssim, end="\n\n")

    print("Ended evaluation!\n")

    # save the results
    eval_f_name = '%s/evaluation/results.json' % opt.log_dir
    plot_dir_name = '%s/evaluation/plots/' % opt.log_dir
    utils.save_eval(eval_f_name, avg_test_mse, avg_test_miou, avg_test_ssim, plot_dir=plot_dir_name)


if __name__ == '__main__':
    main()
