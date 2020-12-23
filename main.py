from sklearn.metrics import confusion_matrix, accuracy_score

from ratings_classifier import RatingsClassifier
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
	# Remove ticker and description - don't need these for classification
	# Also drop any na columns
	data = pd.read_csv("RatingsAndFundamentals.csv").dropna()
	data = data.drop(columns=['RTG_SP_LT_LC_ISSUER_CREDIT', 'Ticker', 'Name']).dropna()

	print("Total Observations: {0}".format(len(data)))

	# Normalise the data using MinMaxScalar
	norm = MinMaxScaler().fit(data)
	data_scaled = pd.DataFrame(norm.transform(data))
	data_scaled.columns = data.columns

	# Correlation between all features
	plt.figure(figsize=(20, 20))
	g = sns.heatmap(data_scaled.corr(), annot=True, cmap="RdYlGn")
	plt.show()

	# Split data into train and test sets
	Y_scaled = data.OurRating
	X_scaled = data_scaled.drop(columns=['OurRating'])
	X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.3)

	# Fit the data using the training data set
	ratings_class = RatingsClassifier(display_plots=False)
	ratings_class.fit(X_train, Y_train)

	# Store the results from fitting and using CV
	print(ratings_class.results.head(1000))
	ratings_class.results.to_csv("ModelCrossValidationResults.csv")

	# Run the prediction using the best model, see ratings_classifier.py for details
	Y_predicted = ratings_class.predict(X_test)

	# Plot and save the results
	cm = confusion_matrix(Y_test, Y_predicted)
	sns.heatmap(pd.DataFrame(cm), annot=True, annot_kws={"size": 16}, )
	plt.show()
	# Show the Accuracy Score on the test set
	print(accuracy_score(Y_test, Y_predicted))
