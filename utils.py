import random

from dataloader.dataset import SequencedDataset
import torch
from torch.autograd import Variable
import numpy as np
import imageio
import json
import os
import matplotlib.pyplot as plt


def _split_samples_train_test(samples, list_of_test_towns):
    test_samples = list()
    train_samples = list()

    for sample in samples:
        town_name, _ = sample
        if any(test_town in town_name for test_town in list_of_test_towns):
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, test_samples


def load_dataset(opt):
    # can be all(100%), small(20%), tiny(2%)
    # samples is none only if user wants to train without extracting the dataset, possibly on a tiny dataset
    train_is_test = opt.train_is_test
    size = opt.train_on
    if size == "all":
        size = 1
    elif size == "small":
        size = 0.2
    elif size == "tiny":
        size = 0.02

    train_data = SequencedDataset(opt.data_root, opt.n_past + opt.n_future, opt.n_past, opt.channels == 1)
    all_samples = train_data.samples

    if train_is_test:
        print("Train is Test! Train and test set will be the same.")
        print("So, no town splits...")
        # then the samples will not be split wrt towns
        interested_samples = all_samples[:int(len(all_samples) * size)]
        train_data.samples = interested_samples
        test_data = SequencedDataset(opt.data_root, opt.n_past + opt.n_future, opt.n_past, opt.channels == 1,
                                     samples=interested_samples)

    # if you know which towns will be in the test set
    elif opt.test_towns is not None:
        print(f"Test towns are provided as: {opt.test_towns}")
        print("Splitting accordingly...")
        # split all data into train and test according to the test towns specified in opt
        train_samples, test_samples = _split_samples_train_test(all_samples, opt.test_towns)
        random.seed(447)
        # shuffle the lists so that if the model is not trained on the full set,
        # the sample distribution across datasets are balanced
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        # take the specified portion
        train_samples = train_samples[:int(len(train_samples) * size)]
        test_samples = test_samples[:int(len(test_samples) * size)]
        print(f"Total number of training sequences: {len(train_samples)}")
        print(f"Total number of testing sequences: {len(test_samples)}")
        # assign the samples to the datasets
        train_data.samples = train_samples
        test_data = SequencedDataset(opt.data_root, opt.n_past + opt.n_future, opt.n_past, opt.channels == 1,
                                     samples=test_samples)
    else:
        # split wrt to the proportion specified
        # do not shuffle this time, to not copy from train set
        # still as the towns are shared this method is not good
        print("Train is not test, and no test towns provided.")
        print("Splitting only according to the proportions...")
        train_prop = opt.train_test_ratio / (opt.train_test_ratio + 1)
        interested_samples = all_samples[:int(len(all_samples) * size)]
        train_last_idx = int(train_prop * len(interested_samples))
        train_data.samples = interested_samples[:train_last_idx]
        test_data = SequencedDataset(opt.data_root, opt.n_past + opt.n_future, opt.n_past, opt.channels == 1,
                                     samples=interested_samples[train_last_idx:])

    return train_data, test_data


def normalize_data(dtype1, dtype2, sequence):
    # squeeze images into 0-1
    frames = sequence["frames"]
    actions = sequence["actions"]
    frames /= 256.0
    return frames.type(dtype1), actions.type(dtype2)


# #############################################!!!  !!!##############################################
def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def init_weights(m):
    classname = m.__class__.__name__
    if classname == 'ConvLSTMCell' or classname == 'ConvLSTM' or classname == 'ConvGaussianLSTM':
        return
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# #############################################!!!  !!!##############################################


def save_loss(loss_f_name, train_minibatch_losses, eval_batch_losses):
    results = {}
    num_epochs = len(train_minibatch_losses)

    for epoch in range(num_epochs):
        epoch_data = {
            "avg_train_mse": 0,
            "avg_train_kld": 0,
            "avg_eval_mse": float(eval_batch_losses[epoch]) if epoch < len(eval_batch_losses) else None,
            "train_minibatch_mses": [],
            "train_minibatch_klds": []
        }

        total_mse = 0
        total_kld = 0
        num_batches = len(train_minibatch_losses[epoch])

        for batch in train_minibatch_losses[epoch]:
            mse, kld = batch
            total_mse += mse
            total_kld += kld
            epoch_data["train_minibatch_mses"].append(float(mse))
            epoch_data["train_minibatch_klds"].append(float(kld))

        epoch_data["avg_train_mse"] = float(total_mse / num_batches)
        epoch_data["avg_train_kld"] = float(total_kld / num_batches)

        results[f"epoch_{epoch + 1}"] = epoch_data

    os.makedirs(os.path.dirname(loss_f_name), exist_ok=True)

    with open(loss_f_name, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Loss data saved to {loss_f_name}")


def save_eval(f_name, mse, miou, ssim, plot_dir=None):
    results = {"MSE": mse,
               "MIOU": miou,
               "SSIM": ssim}

    with open(f_name, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Eval data saved to {f_name}")

    if plot_dir is not None:
        sequence_index = list(range(1, len(results["MSE"]) + 1))

        plt.figure()
        plt.plot(sequence_index, results["MSE"], marker='o', linestyle='-', color='blue')
        plt.title("MSE Across Frames")
        plt.xlabel("Sequence Index")
        plt.ylabel("Score")
        plt.xticks(sequence_index)  # Set x-ticks to be explicit sequence indices
        plt.savefig(os.path.join(plot_dir, "mse_across_frames.png"))
        plt.close()

        plt.figure()
        plt.plot(sequence_index, results["MIOU"], marker='o', linestyle='-', label='MIOU', color='green')
        plt.plot(sequence_index, results["SSIM"], marker='o', linestyle='-', label='SSIM', color='red')
        plt.title("MIOU and SSIM Across Frames")
        plt.xlabel("Sequence Index")
        plt.ylabel("Score")
        plt.xticks(sequence_index)  # Set x-ticks to be explicit sequence indices
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "miou_ssim_across_frames.png"))
        plt.close()


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        new_image = torch.zeros(img.shape[0], img.shape[1], 3)
        # complete other channels
        for i in range(img.shape[2]):
            new_image[:, :, 2-i] = img[:, :, -1 - i]
        new_image = (new_image.numpy() * 255).astype(np.uint8)
        images.append(new_image)
    imageio.mimsave(filename, images, duration=duration)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    from PIL import Image
    tensor = (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

