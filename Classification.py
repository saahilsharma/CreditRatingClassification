import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import copy

# Machine learning
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix  # for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn.decomposition import PCA

class RatingsClassification(object):
	def __init__(self, test_size=0.25):
		self.methods = ['LR', 'LDA', 'DT', 'RF', 'LinSVM']
		self.variable_sel_methods = ['Lasso','PCA','Correlation']
		self.scores = ['Accuracy', 'Precision', 'Recall', 'F1']
		self.results:DataFrame = pd.DataFrame(index=pd.MultiIndex.from_product([self.methods, self.variable_sel_methods]), columns=self.scores)
		self.test_size = test_size
		self.best_estimator = dict()

	def run_classification(self, X_data, Y_data):
		X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=self.test_size)

		### Lasso ###
		X_reduced = self.__reduce_with_lasso(X_train, Y_train)

		lr_grid = self.__run_lr(X_reduced,Y_train)
		lda_grid = self.__run_lda(X_train,Y_train)
		dt_grid = self.__run_dt(X_train,Y_train)
		rf_grid = self.__run_rf(X_train,Y_train)
		svm_grid = self.__run_svm(X_train,Y_train)

		### PCA ###
		lr_grid_pca = self.__run_lr(X_reduced, Y_train, use_pca=True)
		lda_grid_pca = self.__run_lda(X_train, Y_train, use_pca=True)
		dt_grid_pca = self.__run_dt(X_train, Y_train, use_pca=True)
		rf_grid_pca = self.__run_rf(X_train, Y_train, use_pca=True)
		svm_grid_pca = self.__run_svm(X_train, Y_train, use_pca=True)

		###

	# def fit_model(self, grid, method, sel_method):
	#  	self.results.loc[method][sel_method]['Accuracy'] = grid.best_estimator_['accuracy'].mean()
	# 	self.results.loc[method]['Accuracy'] = scores['test_accuracy'].mean()
	#   	self.results.loc[method]['Precision'] = scores['test_precision'].mean()
	#   	self.results.loc[method]['Recall'] = scores['test_recall'].mean()
	#   	self.results.loc[method]['F1'] = scores['test_f1'].mean()

	def __reduce_with_lasso(self, X, Y):
		lasso = SelectFromModel(LassoCV(random_state=42, tol=0.01))
		lasso.fit(X,Y)
		retained_feats = X.columns[(lasso.estimator_.coef_ != 0).ravel().tolist()]
		print(retained_feats)
		return lasso.transform(X)

	def __run_lr(self, X, Y, use_pca=False):
		lr = LogisticRegression(random_state=42)
		C = [0.01, 0.1, 1, 10]
		if not(use_pca):
			params = dict(lr__C=C)
			pipe = Pipeline(steps=[('lr', lr)])
		else:
			n_components = list(range(1, X.shape[1] + 1, 1))
			params = dict(dim_reduction__n_components=n_components, lr__C=C)
			pipe =  Pipeline([('dim_reduction', PCA(random_state=42)),
		                     ('lr', lr)]
		                    )
		return self.__find_best_estimator(pipe, params, X, Y)

	def __run_lda(self, X, Y, use_pca=False):
		lda = LinearDiscriminantAnalysis()
		if not(use_pca):
			params = dict()
			pipe = Pipeline(steps=[('lda', lda)])
		else:
			n_components = list(range(1, X.shape[1] + 1, 1))
			params = dict(dim_reduction__n_components=n_components)
			pipe = Pipeline([('dim_reduction', PCA(random_state=42)),
			                     ('lda', lda)]
			                    )

		return self.__find_best_estimator(pipe, params, X, Y)

	def __run_dt(self, X, Y, use_pca=False):
		n_leaf_nodes = range(2,40)
		dt = tree.DecisionTreeClassifier(random_state=42)
		if not(use_pca):
			params = dict(dt__max_leaf_nodes=n_leaf_nodes)
			pipe = Pipeline(steps=[('dt', dt)])
		else:
			n_components = list(range(1, X.shape[1] + 1, 1))
			params = dict(dim_reduction__n_components=n_components, dt__max_leaf_nodes=n_leaf_nodes)
			pipe = Pipeline([('dim_reduction', PCA(random_state=42)),
		                 ('dt', tree.DecisionTreeClassifier(random_state=42))]
		                )

		return self.__find_best_estimator(pipe, params, X, Y)

	def __run_rf(self, X, Y, use_pca=False):
		max_features = ['auto', 'sqrt', 'log2']
		n_estimators = range(200, 400, 50)
		max_depth = [4, 5, 6, 7, 8]
		criterion = ['gini', 'entropy']
		if not(use_pca):
			params = dict(rf__max_features=max_features, rf__n_estimators=n_estimators,
			              rf__max_depth=max_depth, rf__criterion=criterion)
			pipe = Pipeline([('rf', RandomForestClassifier(random_state=42))])
		else:
			n_components = list(range(1, X.shape[1] + 1, 1))
			params = dict(dim_reduction__n_components=n_components, rf__max_features=max_features,
			              rf__n_estimators=n_estimators, rf__max_depth=max_depth, rf__criterion=criterion)
			pipe = Pipeline([('dim_reduction', PCA(random_state=42)),
			                 ('rf', RandomForestClassifier(random_state=42))]
			                )

		return self.__find_best_estimator(pipe, params, X, Y)

	def __run_svm(self, X, Y, use_pca=False):
		C = [0.001,0.01, 0.1, 1, 10]
		gammas = [0.001, 0.01, 0.1, 1]
		kernels = ['linear','poly','rbf']
		svm = SVC()
		if not(use_pca):
			params = dict(svm__C=C, svm__gamma=gammas,svm__kernel=kernels)
			pipe = Pipeline(steps=[('svm', svm)])
		else:
			n_components = list(range(1, X.shape[1] + 1, 1))
			params = dict(dim_reduction__n_components=n_components, svm__C=C, svm__gamma=gammas,svm__kernel=kernels)
			pipe = Pipeline([('dim_reduction', PCA(random_state=42)),
		                 ('svm', SVC(random_state=42))]
		                )

		return self.__find_best_estimator(pipe, params, X, Y)

	def __find_best_estimator(self, pipe, param_grid, X, Y):
		grid = GridSearchCV(pipe, cv=10, n_jobs=2, param_grid=param_grid, scoring=["accuracy", "f1", "precision", "recall"],
		                    refit='accuracy', return_train_score=True)
		grid.fit(X,Y)
		#print(grid.cv_results_['mean_test_f1'])
		print(grid.cv_results_['rank_test_accuracy'])
		#print(grid.best_estimator_['dim_reduction'].components_)
		#print(pd.DataFrame(grid.best_estimator_['dim_reduction'].components_, columns=X.columns))
		self.__plot_results(grid)
		return grid

	def __plot_results(self, grid):
		scoring=["accuracy", "f1", "precision", "recall"]
		results = grid.cv_results_
		plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
		          fontsize=16)

		plt.xlabel("param_lr__C")
		plt.ylabel("Score")

		ax = plt.gca()
		#ax.set_xlim(0, 100)
		#ax.set_ylim(0.73, 1)

		# Get the regular numpy array from the MaskedArray
		X_axis = np.array(results['param_lr__C'].data, dtype=float)

		for scorer, color in zip(sorted(scoring), ['g', 'k']):
			for sample, style in (('train', '--'), ('test', '-')):
				sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
				sample_score_std = results['std_%s_%s' % (sample, scorer)]
				ax.fill_between(X_axis, sample_score_mean - sample_score_std,
				                sample_score_mean + sample_score_std,
				                alpha=0.1 if sample == 'test' else 0, color=color)
				ax.plot(X_axis, sample_score_mean, style, color=color,
				        alpha=1 if sample == 'test' else 0.7,
				        label="%s (%s)" % (scorer, sample))

			best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
			best_score = results['mean_test_%s' % scorer][best_index]

			# Plot a dotted vertical line at the best score for that scorer marked by x
			ax.plot([X_axis[best_index], ] * 2, [0, best_score],
			        linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

			# Annotate the best score for that scorer
			ax.annotate("%0.2f" % best_score,
			            (X_axis[best_index], best_score + 0.005))

		plt.legend(loc="best")
		plt.grid(False)
		plt.show()




if __name__ == "__main__":
	data = pd.read_csv("RatingsAndFundamentals.csv").dropna()
	data = data.drop(columns=['RTG_SP_LT_LC_ISSUER_CREDIT', 'Ticker', 'Name']).dropna()

	# fit scaler on data
	norm = MinMaxScaler().fit(data)
	#norm = StandardScaler().fit(data)

	# transform training data
	data_scaled = pd.DataFrame(norm.transform(data))
	data_scaled.columns = data.columns

	Y_scaled = data.OurRating
	X_scaled = data_scaled.drop(columns=['OurRating'])

	ratings_class = RatingsClassification(test_size=0.3)
	ratings_class.run_classification(X_scaled, Y_scaled)

	print(ratings_class.results.head(10))

#
# X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.3)
#
# lr_pipe = Pipeline([('pca', PCA(n_components=3)),
#                     ('clf', LogisticRegression())])
#
# scores = cross_validate(estimator=lr_pipe, X=X_train, y=Y_train, cv=10, scoring=["accuracy","f1","precision","recall"], return_train_score=True)
# print(scores)
#
# log_class = LogisticRegression()
# lda_class = LinearDiscriminantAnalysis()
# qda_class = QuadraticDiscriminantAnalysis()
# rf_class = RandomForestClassifier(n_estimators=10)
# svm_class = LinearSVC()
# dt_class = tree.DecisionTreeClassifier()
# # Tree with Pruning
# parameters = {'max_leaf_nodes': range(2, 40)}
# cv_tree = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=10, scoring='accuracy')
#
# print("Random Forests: ")
# accuracy = cross_val_score(rf_class, X_scaled, Y_scaled, scoring='accuracy', cv=10).mean() * 100
# print("Accuracy of Random Forests is: ", accuracy)
#
# print("\n\nSVM:")
# accuracy = cross_val_score(svm_class, X_scaled, Y_scaled, scoring='accuracy', cv=10).mean() * 100
# print("Accuracy of SVM is: ", accuracy)
#
# print("\n\nDecision Tree:")
# accuracy = cross_val_score(dt_class, X_scaled, Y_scaled, scoring='accuracy', cv=10).mean() * 100
# print("Accuracy of Decision Tree is: ", accuracy)
#
# print("\n\nDecision Tree With Pruning:")
# cv_tree.fit(X=X_scaled, y=Y_scaled)
# accuracy = cv_tree.score(X_scaled,  Y_scaled).mean() * 100
# print("Accuracy of Decision Tree with Pruning is: ", accuracy)
#
# print("\n\nLogistic Regression:")
# accuracy = cross_val_score(log_class, X_scaled, Y_scaled, scoring='accuracy', cv=10).mean() * 100
# print("Accuracy of Logistic Regression is: ", accuracy)
#
# print("\n\nLinearDiscriminantAnalysis:")
# accuracy = cross_val_score(lda_class, X_scaled, Y_scaled, scoring='accuracy', cv=10).mean() * 100
# print("Accuracy of LinearDiscriminantAnalysis is: ", accuracy)
#
# print("\n\nQuadraticDiscriminantAnalysis:")
# accuracy = cross_val_score(qda_class, X_scaled, Y_scaled, scoring='accuracy', cv=10).mean() * 100
# print("Accuracy of QuadraticDiscriminantAnalysis is: ", accuracy)
#
# model = ExtraTreesClassifier()
# model.fit(X_scaled, Y_scaled)
# # plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X_scaled.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()
#
# corrmat = data_scaled.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20, 20))
# # plot heat map
# g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
# plt.show()
#
# pca = PCA(n_components=3)
# fit = pca.fit(X_scaled)
# print(fit.explained_variance_ratio_)
# feat_importances = pd.Series(fit.components_, index=X_scaled.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()
# print(fit.components_)
