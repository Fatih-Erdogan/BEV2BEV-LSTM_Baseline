import os
import json
import matplotlib.pyplot as plt
from types import SimpleNamespace


def plot_loss_curve(path_file, out_path, train_mse=True, test_mse=True, train_kld=False, show=False):
    with open(path_file, "r") as f:
        data = json.load(f)

    epochs = range(1, len(data.keys()) + 1)
    avg_train_mse = []
    avg_train_kld = []
    avg_eval_mse = []

    for epoch in epochs:
        epoch_data = data[f"epoch_{epoch}"]
        avg_train_mse.append(epoch_data['avg_train_mse'])
        avg_train_kld.append(epoch_data['avg_train_kld'])
        avg_eval_mse.append(epoch_data['avg_eval_mse'])

    plt.figure(figsize=(10, 5))
    if train_mse:
        plt.plot(epochs, avg_train_mse, label='Average Train MSE', marker='o')
    if train_kld:
        plt.plot(epochs, avg_train_kld, label='Average Train KLD', marker='o')
    if test_mse:
        plt.plot(epochs, avg_eval_mse, label='Average Eval MSE', marker='o')

    plt.title('Training and Evaluation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    if show:
        plt.show()


def main(opt):
    assert opt.model_dir != ""
    loss_file_name = os.path.join(opt.model_dir, "loss/out.json")
    out_out_folder_name = os.path.join(opt.model_dir, "plots/")
    os.makedirs(out_out_folder_name, exist_ok=True)

    out_file_name = os.path.join(out_out_folder_name, "training_curve_wo_kld.png")

    plot_loss_curve(loss_file_name, out_file_name, train_mse=True, test_mse=True, train_kld=False, show=False)


if __name__ == '__main__':
    with open("config.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)
    main(opt)
