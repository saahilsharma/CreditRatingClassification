import os
import json

import numpy as np
import pickle


class DatasetGenerator(object):
    """
    Data Generator is used to generate simulated data for
    testing lasso and elastic net
    """

    def __init__(self, n_train, n_validation, n_test):
        self.n_train_ = n_train
        self.n_validation_ = n_validation
        self.n_test_ = n_test

        self.n_total_ = n_train + n_validation + n_test

        self.beta_ = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])


    def random_dataset(self, n_dims, sigma, corr_factor):
        """
        Generate a random dataset (no pre-specified correlation)

        @param n_dims (int): dimension of each sample
        @param sigma (float): scale of the noise
        @param corr_factor: correlation factor of X
        @returns (a dictionary of np.array)
            {
                "train": np.array of shape (n_train, n_dims),
                "validation": np.array of shape (n_validation, n_dims),
                "test": np.array of shape (n_test, n_dims),
            }
        """
        # samples_x = np.random.rand(self.n_total_, n_dims)
        samples_x = self.sample_corr_x_(n_dims, factor=corr_factor)

        samples_y = self.calc_y(samples_x, sigma)

        train_x = samples_x[:self.n_train_, :]
        validation_x = samples_x[self.n_train_:self.n_train_+self.n_validation_, :]
        test_x = samples_x[-self.n_test_:, :]

        train_y = samples_y[:self.n_train_]
        validation_y = samples_y[self.n_train_:self.n_train_+self.n_validation_]
        test_y = samples_y[-self.n_test_:]
        
        dataset = {
            "train_x": train_x,
            "train_y": train_y,
            "validation_x": validation_x,
            "validation_y": validation_y,
            "test_x": test_x,
            "test_y": test_y
        }

        assert(train_x.shape == (self.n_train_, n_dims))
        assert(validation_x.shape == (self.n_validation_, n_dims))
        assert(test_x.shape == (self.n_test_, n_dims))
        assert(train_y.shape == (self.n_train_,))
        assert(validation_y.shape == (self.n_validation_,))
        assert(test_y.shape == (self.n_test_,))

        return dataset


    def sample_corr_x_(self, n_dims, factor):
        cov = np.zeros((n_dims, n_dims))

        for i in range(n_dims):
            for j in range(i, n_dims):
                cov[j][i] = cov[i][j] = factor ** (j - i)
        
        x = np.random.multivariate_normal(mean=[0] * n_dims, cov=cov, size=self.n_total_)
        assert(x.shape == (self.n_total_, n_dims))

        return x
    
    
    def calc_y(self, X, sigma):
        noise = np.random.multivariate_normal(
            mean=[0] * self.n_total_,
            cov=np.diag([1] * self.n_total_),
        )
        Y = np.dot(X, self.beta_) + sigma * noise
        
        assert(Y.shape == (self.n_total_,))

        return Y


def dump_dataset(dataset, fileout):
    dataset_json = {k: v.tolist() for k, v in dataset.items()}

    with open(fileout, "w") as fout:
        json.dump(dataset_json, fout)
    


def generate_dataset():
    n_dims = 8
    n_train = 50
    n_validation = 50
    n_test = 200
    n_datasets = 50
    output_dir = "dataset"
    sigma = 0.1
    corr_factor = 0.1

    # generate outpud directory
    os.system("mkdir -p {}".format(output_dir))

    dg = DatasetGenerator(n_train, n_validation, n_test)

    # generate random dataset
    for i in range(n_datasets):
        dataset_name = "sigma_{}_corr_{}_{}.json".format(sigma, corr_factor, i)
        fileout = "{}/{}".format(output_dir, dataset_name)

        dataset = dg.random_dataset(n_dims, sigma, corr_factor)

        dump_dataset(dataset, fileout)
        print("[INFO] Dataset is saved at: {}".format(fileout))


if __name__ == "__main__":
    generate_dataset()