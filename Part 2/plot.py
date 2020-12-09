import matplotlib.pyplot as plt
import numpy as np



def box_plot(lasso_error, lasso_nonzero, elastic_error, elastic_nonzero, title):
    file_name = "_".join(title.split(' '))

    fig = plt.figure(figsize =(10, 7)) 
    ax = fig.add_subplot(111)
    ax.boxplot([lasso_error, elastic_error], notch=True, showfliers=False)
    ax.set_xticklabels(["Lasso", "Elastic Net"], fontsize=20)
    ax.set_xlabel("Model", fontsize=20)
    ax.set_ylabel("Averaged MSE", fontsize=20)
    ax.set_title("{}\nAverage Mean-Square-Error".format(title), fontsize=20)
    plt.tick_params(labelsize=16)
    
    plt.savefig("mse_{}.png".format(file_name))
    plt.show()

    plt.clf()

    fig = plt.figure(figsize =(10, 7)) 
    ax = fig.add_subplot(111)
    ax.boxplot([lasso_nonzero, elastic_nonzero], notch=True, showfliers=False)
    ax.set_xticklabels(["Lasso", "Elastic Net"], fontsize=20)
    ax.set_xlabel("Model", fontsize=20)
    ax.set_ylabel("Averaged # of nonzero coef", fontsize=20)
    ax.set_title("{}\nAveraged number of nonzero coef".format(title), fontsize=20)
    plt.tick_params(labelsize=16)

    plt.savefig("nonzero_{}.png".format(file_name))
    plt.show()