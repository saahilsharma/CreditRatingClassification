import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import quandl

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
import copy

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix  # for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

if __name__ == "__main__":
	data = pd.read_csv("RatingsAndFundamentals.csv")
	print(data.head())

	# plt.scatter(result['fcf'], result['OurRating'], c='g', s=4)
	# plt.show()
	# plt.scatter(result['eps'], result['OurRating'], c='r', s=4)
	# plt.show()
	# plt.scatter(result['roe'], result['OurRating'], c='b', s=4)
	# plt.show()
	# plt.scatter(result['roa'], result['OurRating'], c='y', s=4)
	# plt.show()
	# plt.scatter(result['ebit'], result['OurRating'], c='b', s=4)
	# plt.show()
	plt.scatter(data['OurRating'], data['netinc'], c='orange', s=4)
	plt.show()