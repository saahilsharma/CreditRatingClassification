import numpy as np

from lasso import LassoModel
from elastic_net import ElasticNet
from utils import load_json
from utils import evaluate, count_nonzero_coef
from plot import box_plot



DATASET_1 = [
    "dataset/sigma_0_corr_0.8_{}.json".format(i) for i in range(50)
]

DATASET_2 = [
    "dataset/sigma_0.1_corr_0.1_{}.json".format(i) for i in range(50)
]

DATASET_3 = [
    "dataset/sigma_3_corr_0.5_{}.json".format(i) for i in range(50)
]

DATASETS = {
    "sigma_0_corr_0.8": DATASET_1,
    "sigma_0.1_corr_0.1": DATASET_2,
    "sigma_3_corr_0.5": DATASET_3
}


class Tester(object):

    def __init__(self):
        pass

    def test_model(self, model_name, dataset):
        model = self.gen_model_(model_name)

        error_hist = []
        nonzero_hist = []
        for dataset_file in dataset:
            train_x, train_y, validation_x, validation_y, test_x, test_y = self.load_dataset(dataset_file)

            min_error = 999
            selected_lambda = 0.0

            # select parameter via validation dataset
            for lambda_param in range(1, 10):
                lambda_param /= 10
                model.fit(train_x, train_y, lambda_param=lambda_param)
                predicted_y = model.predict(validation_x)
                error = evaluate(predicted_y, validation_y)

                if error < min_error:
                    min_error = error
                    selected_lambda = lambda_param
            
            # testing the model
            model.fit(train_x, train_y, lambda_param=selected_lambda)
            predicted_y = model.predict(test_x)
            error = evaluate(predicted_y, test_y)
            n_nonzero = count_nonzero_coef(model.beta_)

            # print("Mean Squared Error: {}".format(error))
            error_hist.append(error)
            nonzero_hist.append(n_nonzero)
        
        print("Averaged MSE: {}".format(np.average(error_hist)))
        print("Averaged nonzero coefficients: {}".format(np.average(nonzero_hist)))

        return error_hist, nonzero_hist

    
    def test_model_lambda_2(self, model_name, dataset):
        model = self.gen_model_(model_name)

        error_hist = []
        nonzero_hist = []
        for dataset_file in dataset:
            train_x, train_y, validation_x, validation_y, test_x, test_y = self.load_dataset(dataset_file)

            best_l1, best_l2 = None, None
            min_error = 9999

            for l1 in range(1, 11, 2):
                l1 /= 10.0
                for l2 in range(1, 11, 2):
                    l2 /= 10.0

                    model.fit(train_x, train_y, lambda_1=l1, lambda_2=l2)
                    predicted_y = model.predict(validation_x)
                    error = evaluate(predicted_y, validation_y)

                    if error < min_error:
                        min_error = error
                        best_l1 = l1
                        best_l2 = l2
            
            # testing the model
            model.fit(train_x, train_y, lambda_1=best_l1, lambda_2=best_l2)
            predicted_y = model.predict(test_x)
            error = evaluate(predicted_y, test_y)
            n_nonzero = count_nonzero_coef(model.beta_)

            # print("Mean Squared Error: {}".format(error))
            error_hist.append(error)
            nonzero_hist.append(n_nonzero)
        
        print("Averaged MSE: {}".format(np.average(error_hist)))
        print("Averaged nonzero coefficients: {}".format(np.average(nonzero_hist)))

        return error_hist, nonzero_hist


    def gen_model_(self, model_name):
        if model_name == "lasso":
            model = LassoModel()
        elif model_name == "elastic":
            model = ElasticNet()

        return model


    def load_dataset(self, dataset_file):
        dataset = load_json(dataset_file)
        
        train_x = np.array(dataset["train_x"])
        train_y = np.array(dataset["train_y"])

        validation_x = np.array(dataset["validation_x"])
        validation_y = np.array(dataset["validation_y"])

        test_x = np.array(dataset["test_x"])
        test_y = np.array(dataset["test_y"])

        return train_x, train_y, validation_x, validation_y, test_x, test_y


def launch_tester():
    tester = Tester()
    
    lasso_error_hist, lasso_nonzero_hist = tester.test_model(
        model_name="lasso", dataset=DATASETS["sigma_3_corr_0.5"]
    )
    elastic_error_hist, elastic_nonzero_hist = tester.test_model_lambda_2(
        model_name="elastic", dataset=DATASETS["sigma_3_corr_0.5"]
    )

    box_plot(
        lasso_error_hist,
        lasso_nonzero_hist,
        elastic_error_hist,
        elastic_nonzero_hist,
        "High Noise Medium Correlation"
    )

if __name__ == "__main__":
    launch_tester()