import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Data:
    def __init__(self, data):
        self.data = data
        self.var = 1

class Dataset:
    def __init__(self):
        self.data_square = self._read_data()
        self.data = self._square_to_vector()
        self.var = 1

    def _read_data(self):
        data_folder_path = "./data"
        np_folder_name = "MNIST_numpy"
        np_folder_path = os.path.join(data_folder_path, np_folder_name)
        np_file_path = os.path.join(np_folder_path, "dataset.npy")
        if not os.path.exists(np_file_path):
            train_dataset = torchvision.datasets.MNIST(root=data_folder_path, train=True, transform=transforms.ToTensor(), download=True)
            dataset = []
            for data in train_dataset:
                if data[1] == 2:
                    dataset.append(data[0])
            dataset = torch.stack(dataset).numpy()
            os.mkdir(np_folder_path)
            np.save(np_file_path, dataset)
        else:
            dataset = np.load(np_file_path)
        return dataset
    
    def _square_to_vector(self):
        return np.reshape(self.data_square, (self.data_square.shape[0], -1))

    def get_dimension(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        return Data(self.data[idx])

class NormalDistribution:
    def __init__(self, ave, var, N=None, M=None, D=None, Y=None):
        self.ave_pre = ave
        self.var_pre = var
        self.ave = ave.copy()
        self.var = var.copy()
        self.N = N
        self.M = M
        self.D = D
        self.Y = Y

    def E(self):
        return self.ave
    
    def E_one(self, ind):
        return self.ave[ind]
    
    def E_matrix(self):
        return self.ave[:, np.newaxis] @ self.ave[np.newaxis, :] + self.var

class Coefficient(NormalDistribution):
    def __init__(self, ave, var, N, Y):
        super().__init__(ave, var, N=N, Y=Y)

    def update_ave(self, mu, X, j):
        sum1 = 0
        for i in range(self.N):
            sum1 += (self.Y.data[i, j] - mu.E_one(j)) * X.E(i)
        self.ave = 1 / self.Y.var * self.var @ sum1

    def update_var(self, X):
        sum1 = 0
        for i in range(self.N):
            sum1 += X.E_matrix(i)
        self.var = np.linalg.inv(1 / self.Y.var * sum1 + np.linalg.inv(self.var_pre))

    def update(self, mu, X, j):
        self.update_var(X)
        self.update_ave(mu, X, j)
        return self

class CoefficientAll:
    def __init__(self, dists):
        self.dists = dists
    
    def E(self):
        return np.array([obj.E() for obj in self.dists]).T

    def E_T(self):
        return np.array([obj.E() for obj in self.dists])
    
    def E_each_square(self, ind):
        return self.dists[ind].E_matrix()
    
    def update(self, mu, X):
        self.dists = [obj.update(mu, X, j) for j, obj in enumerate(self.dists)]

class Mean(NormalDistribution):
    def __init__(self, ave, var, N, D, Y):
        super().__init__(ave, var, N=N, D=D, Y=Y)

    def update_ave(self, W, X):
        W_T = W.E_T()
        sum1 = 0
        for i in range(N):
            sum1 += self.Y.data[i] - W_T @ X.E(i)
        self.ave =  1 / self.Y.var * self.var @ sum1

    def update_var(self):
        self.var = np.linalg.inv(self.N / self.Y.var * np.eye(self.D) + np.linalg.inv(self.var_pre))

class LatentVariable(NormalDistribution):
    def __init__(self, ave, var, D, M, Y):
        super().__init__(ave, var, D=D, M=M, Y=Y)

    def update_ave(self, W, mu, j):
        return 1 / self.Y.var * self.var @ W.E() @ (self.Y.data[j] - mu.E().shape)

    def update_var(self, W):
        sum1 = 0
        for i in range(self.D):
            sum1 += W.E_each_square(i)
        self.var = np.linalg.inv(1 / self.Y.var * sum1 + np.eye(self.M))

    def update(self, W, mu, j):
        self.update_var(W)
        self.update_ave(W, mu, j)
        return self

class LatentVariableAll:
    def __init__(self, dists):
        self.dists = dists

    def E(self, ind):
        return self.dists[ind].E()
    
    def E_matrix(self, ind):
        return self.dists[ind].E_matrix()
    
    def update(self, W, mu):
        self.dists = [obj.update(W, mu, j) for j, obj in enumerate(self.dists)]

class CalculateParameters:
    def __init__(self, N, M):
        self.dataset = Dataset()
        self.N = N # number of dataset
        self.M = M # number of dimension for latent variable
        self.D = self.dataset.get_dimension() # number of dimension for variable
        self.Y = self.dataset[0:N] # data
        self.P_W = CoefficientAll([Coefficient(np.random.rand(self.M), np.random.rand(self.M, self.M) + np.eye(self.M), N=self.N, Y=self.Y) for _ in range(self.D)])
        self.P_mu = Mean(np.zeros(self.D), np.eye(self.D), N=self.N, D=self.D, Y=self.Y)
        self.P_X = LatentVariableAll([LatentVariable(np.random.rand(self.M), np.random.rand(self.M, self.M) + np.eye(self.M), D=self.D, M=self.M, Y=self.Y) for _ in range(self.N)])

    def update(self, maxiter):
        for i in range(maxiter):
            if i == 0:
                self.P_mu.update_var()
            self.P_mu.update_ave(self.P_W, self.P_X)
            self.P_W.update(self.P_mu, self.P_X)
            self.P_X.update(self.P_W, self.P_mu)

    def restore(self):
        Y_compressed = []
        for i in range(self.N):
            Y_compressed.append(self.P_W.E_T() @ self.P_X.E(i) + self.P_mu.E())
        return self.Y.data, np.stack(Y_compressed)

def save_all(original, compression):
    original = original.reshape(original.shape[0], 28, 28)
    compression = compression.reshape(compression.shape[0], 28, 28)
    os.makedirs("out", exist_ok=True)
    for i, img in enumerate(original):
        img = Image.fromarray((img*256).astype(np.int8), mode="L")
        img.save(f"out/{i:03}_original.jpg")
    for i, img in enumerate(compression):
        img = Image.fromarray((img*256).astype(np.int8), mode="L")
        img.save(f"out/{i:03}_compression.jpg")

if __name__ == "__main__":
    N = 1000 # number of dataset
    M = 30 # number of dimension for latent variable
    maxiter = 20 # number of updating parameters
    calculater = CalculateParameters(N=N, M=M)
    calculater.update(maxiter=maxiter)
    img_original, img_compressed = calculater.restore()
    save_all(img_original, img_compressed)