import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def read_data():
    data_folder_path = "./data"
    np_folder_name = "MNIST_numpy"
    np_folder_path = os.path.join(data_folder_path, np_folder_name)
    np_file_path = os.path.join(np_folder_path, "dataset.npy")
    if not os.path.exists(np_file_path):
        train_dataset = torchvision.datasets.MNIST(root=data_folder_path, train=True, transform=transforms.ToTensor(), download=True)
        dataset = []
        for data in train_dataset:
            dataset.append(data[0])
        dataset = torch.stack(dataset).numpy()
        os.mkdir(np_folder_path)
        np.save(np_file_path, dataset)
    else:
        dataset = np.load(np_file_path)
    return dataset

def calculate_posterior_parameters():
    pass

class Coeffcient():
    def __init__(self, ave, pre):
        self.ave = ave
        self.pre = pre

class Mean():
    def __init__(self, ave, pre):
        self.ave = ave
        self.pre = pre

class LatentVariable():
    def __init__(self, ave, pre):
        self.ave = ave
        self.pre = pre

if __name__ == "__main__":
    N = 100
    dataset = read_data()
    dataset = dataset[0:N]