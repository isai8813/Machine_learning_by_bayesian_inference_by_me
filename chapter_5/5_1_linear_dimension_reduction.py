import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class Dataset:
    def __init__(self):
        self.data_square = self._read_data()
        self.data = self.square_to_vector()

    def _read_data(self):
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
    
    def square_to_vector(self):
        return np.reshape(self.data_square, (self.data_square.shape[0], -1))

    def get_dimension(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        return self.data[idx]

class Coefficient:
    def __init__(self, ave, pre):
        self.ave = ave
        self.pre = pre

    def E(self):
        return self.ave
    
    def E_matrix(self):
        return self.ave[:, np.newaxis] @ self.ave[np.newaxis, :]

class Mean:
    def __init__(self, ave, pre):
        self.ave = ave
        self.pre = pre

    def E(self):
        return self.ave

class LatentVariable:
    def __init__(self, ave, pre):
        self.ave = ave
        self.pre = pre

    def E(self):
        return self.ave
    
    def E_matrix(self):
        return self.ave[:, np.newaxis] @ self.ave[np.newaxis, :]

class CalculateParameters:
    def __init__(self, N, M):
        self.dataset = Dataset()
        self.N = N # number of dataset
        self.M = M # number of dimension for latent variable
        self.D = self.dataset.get_dimension() # number of dimension for variable
        self.Y = self.dataset[0:N] # data

    def calculate_posterior_parameters(self):
        P_W = [Coefficient(torch.zeros(self.M), torch.eye(self.M)) for _ in range(self.D)]
        P_mu = Mean(torch.zeros(self.D), torch.eye(self.D))
        P_X = [LatentVariable(torch.zeros(self.M), torch.eye(self.M)) for _ in range(self.N)]

if __name__ == "__main__":
    N = 100 # number of dataset
    M = 50 # number of dimension for latent variable
    calculater = CalculateParameters(N=N, M=M)