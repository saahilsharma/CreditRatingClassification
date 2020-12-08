import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif

from pca_helper import PCAHelper

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
from enum import Enum
from functools import partial

class RatingsClassification(object):

	class MLMethod(Enum):
		LogisticRegression = 1,
		LDA = 2,
		DecisionTree = 3,
		RandomForest = 4,
		SVM = 5

	class FeatureSelectionMethod(Enum):
		Lasso = 1,
		PCA = 2,
		F_Score = 3

	class Metrics(Enum):
		Mean = 1,
		Std = 2

	class Scores(Enum):
		accuracy = 1,
		precision = 2,
		recall = 3,
		f1 = 4

	def __init__(self, test_size=0.25):
		methods = [e.name for e in RatingsClassification.MLMethod]
		feature_sel_methods = [e.name for e in RatingsClassification.FeatureSelectionMethod]
		metrics = [e.name for e in RatingsClassification.Metrics]
		scores = [e.name for e in RatingsClassification.Scores]

		self.results = pd.DataFrame(index=pd.MultiIndex.from_product([methods, feature_sel_methods, metrics]),
		                            columns=scores)
		self.test_size = test_size
		self.best_estimator = dict()

	def run_classification(self, X_data, Y_data):
		X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=self.test_size)

		### Lasso ###
		X_reduced = self.__reduce_with_lasso(X_train, Y_train)

		lr_grid = self.__run_lr(X_reduced,Y_train, RatingsClassification.FeatureSelectionMethod.Lasso)
		lda_grid = self.__run_lda(X_train,Y_train, RatingsClassification.FeatureSelectionMethod.Lasso)
		dt_grid = self.__run_dt(X_train,Y_train, RatingsClassification.FeatureSelectionMethod.Lasso)
		rf_grid = self.__run_rf(X_train,Y_train, RatingsClassification.FeatureSelectionMethod.Lasso)
		svm_grid = self.__run_svm(X_train,Y_train, RatingsClassification.FeatureSelectionMethod.Lasso)

		### PCA ###
		lr_grid_pca = self.__run_lr(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.PCA)
		lda_grid_pca = self.__run_lda(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.PCA)
		dt_grid_pca = self.__run_dt(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.PCA)
		rf_grid_pca = self.__run_rf(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.PCA)
		svm_grid_pca = self.__run_svm(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.PCA)

		### F-Score ###
		lr_grid_pca = self.__run_lr(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.F_Score)
		lda_grid_pca = self.__run_lda(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.F_Score)
		dt_grid_pca = self.__run_dt(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.F_Score)
		rf_grid_pca = self.__run_rf(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.F_Score)
		svm_grid_pca = self.__run_svm(X_train, Y_train, RatingsClassification.FeatureSelectionMethod.F_Score)

	def __reduce_with_lasso(self, X, Y):
		lasso = SelectFromModel(LassoCV(random_state=42, tol=0.01))
		lasso.fit(X, Y)
		retained_feats = X.columns[(lasso.estimator_.coef_ != 0).ravel().tolist()]
		print(retained_feats)
		return lasso.transform(X)

	def __run_lr(self, X, Y, sel_method: FeatureSelectionMethod):
		lr = LogisticRegression(random_state=42)
		C = [0.01, 0.1, 1, 10]
		params = dict(lr__C=C)
		pipe = Pipeline(steps=[('lr', lr)])
		if sel_method == RatingsClassification.FeatureSelectionMethod.Lasso:
			self.__find_best_estimator(pipe, params, X, Y, RatingsClassification.MLMethod.LogisticRegression, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.PCA:
			self.__run_pca_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.LogisticRegression, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.F_Score:
			self.__run_fscore_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.LogisticRegression, sel_method)

		# self.__plot_results(grid, "lr__C", sel_method + "_LR")
		# if use_pca:
		# 	self.__plot_results(grid, "dim_reduction__n_components", sel_method + "_LR")

	def __run_lda(self, X, Y, sel_method: FeatureSelectionMethod):
		lda = LinearDiscriminantAnalysis()
		params = dict()
		pipe = Pipeline(steps=[('lda', lda)])
		if sel_method == RatingsClassification.FeatureSelectionMethod.Lasso:
			self.__find_best_estimator(pipe, params, X, Y, RatingsClassification.MLMethod.LDA, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.PCA:
			self.__run_pca_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.LDA, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.F_Score:
			self.__run_fscore_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.LDA, sel_method)

		# if use_pca:
		# 	self.__plot_results(grid, "dim_reduction__n_components", sel_method + "_LDA")

	def __run_dt(self, X, Y, sel_method: FeatureSelectionMethod):
		n_leaf_nodes = range(2, 40)
		dt = tree.DecisionTreeClassifier(random_state=42)
		params = dict(dt__max_leaf_nodes=n_leaf_nodes)
		pipe = Pipeline(steps=[('dt', dt)])
		if sel_method == RatingsClassification.FeatureSelectionMethod.Lasso:
			self.__find_best_estimator(pipe, params, X, Y, RatingsClassification.MLMethod.DecisionTree, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.PCA:
			self.__run_pca_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.DecisionTree, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.F_Score:
			self.__run_fscore_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.DecisionTree, sel_method)

		# self.__plot_results(grid, "dt__max_leaf_nodes", sel_method + "_DT")
		# if use_pca:
		# 	self.__plot_results(grid, "dim_reduction__n_components", sel_method + "_DT")

	def __run_rf(self, X, Y, sel_method: FeatureSelectionMethod):
		n_estimators = range(50, 200, 50)
		max_depth = range(1,10,1)
		params = dict(rf__n_estimators=n_estimators, rf__max_depth=max_depth)
		pipe = Pipeline([('rf', RandomForestClassifier(random_state=42))])
		if sel_method == RatingsClassification.FeatureSelectionMethod.Lasso:
			self.__find_best_estimator(pipe, params, X, Y, RatingsClassification.MLMethod.RandomForest, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.PCA:
			self.__run_pca_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.RandomForest, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.F_Score:
			self.__run_fscore_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.RandomForest, sel_method)

		# self.__plot_results(grid, "rf__n_estimators", sel_method + "_RF")
		# self.__plot_results(grid, "rf__max_depth", sel_method + "_RF")
		# if use_pca:
		# 	self.__plot_results(grid, "dim_reduction__n_components", sel_method + "_RF")

	def __run_svm(self, X, Y, sel_method: FeatureSelectionMethod):
		C = [0.001, 0.01, 0.1, 1, 10]
		gammas = [0.001, 0.01, 0.1, 1]
		kernels = ['linear', 'poly', 'rbf']
		svm = SVC()
		params = dict(svm__C=C, svm__gamma=gammas, svm__kernel=kernels)
		pipe = Pipeline(steps=[('svm', svm)])
		if sel_method == RatingsClassification.FeatureSelectionMethod.Lasso:
			self.__find_best_estimator(pipe, params, X, Y, RatingsClassification.MLMethod.SVM, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.PCA:
			self.__run_pca_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.SVM, sel_method)
		elif sel_method == RatingsClassification.FeatureSelectionMethod.F_Score:
			self.__run_fscore_pipeline(pipe, params, X, Y, RatingsClassification.MLMethod.SVM, sel_method)

		# self.__plot_results(grid, "svm__C", sel_method + "_SVM")
		# self.__plot_results(grid, "svm__gamma", sel_method + "_SVM")
		# self.__plot_results(grid, "svm__kernels", sel_method + "_SVM")
		# if use_pca:
		# 	self.__plot_results(grid, "dim_reduction__n_components", sel_method + "_SVM")

	def __run_pca_pipeline(self, pipe: Pipeline, params: dict, X, Y, method: MLMethod, sel_method: FeatureSelectionMethod):
		n_components = list(range(1, X.shape[1] + 1, 1))
		params['dim_reduction__n_components'] = n_components
		pipe.steps.insert(0, ('dim_reduction', PCA(random_state=42)))
		self.__find_best_estimator(pipe, params, X, Y, method, sel_method, refit_method=PCAHelper.best_low_complexity)

	def __run_fscore_pipeline(self, pipe: Pipeline, params: dict, X, Y, method: MLMethod, sel_method: FeatureSelectionMethod):
		n_components = list(range(1, X.shape[1] + 1, 1))
		params['dim_reduction__k'] = n_components
		pipe.steps.insert(0, ('dim_reduction', SelectKBest(f_classif)))
		self.__find_best_estimator(pipe, params, X, Y, method, sel_method, refit_method=partial(PCAHelper.best_low_complexity,param='dim_reduction__k'))

	def __find_best_estimator(self, pipe, param_grid, X, Y, ml_method: MLMethod, feat_sel_method: FeatureSelectionMethod, refit_method="accuracy"):
		grid = GridSearchCV(pipe, cv=10, n_jobs=-1, param_grid=param_grid,
		                    scoring=[s.name for s in RatingsClassification.Scores],
		                    refit=refit_method, return_train_score=True)
		grid.fit(X, Y)
		results = grid.cv_results_
		print(grid.best_params_)
		method = ml_method.name
		sel_method = feat_sel_method.name
		self.results.loc[method, sel_method, 'Mean']['accuracy'] = results['mean_test_accuracy'][grid.best_index_]
		self.results.loc[method, sel_method, 'Mean']['precision'] = results['mean_test_precision'][grid.best_index_]
		self.results.loc[method, sel_method, 'Mean']['recall'] = results['mean_test_recall'][grid.best_index_]
		self.results.loc[method, sel_method, 'Mean']['f1'] = results['mean_test_f1'][grid.best_index_]

		self.results.loc[method, sel_method, 'Std']['accuracy'] = results['std_test_accuracy'][grid.best_index_]
		self.results.loc[method, sel_method, 'Std']['precision'] = results['std_test_precision'][grid.best_index_]
		self.results.loc[method, sel_method, 'Std']['recall'] = results['std_test_recall'][grid.best_index_]
		self.results.loc[method, sel_method, 'Std']['f1'] = results['std_test_f1'][grid.best_index_]
		return grid

	def __plot_results(self, grid, param, method):
		scoring = ["accuracy"]  # , "f1", "precision", "recall"]
		results = grid.cv_results_
		plt.title(method + " with GridSearchCV", fontsize=16)

		plt.xlabel("param")
		plt.ylabel("Score")

		ax = plt.gca()
		# Get the regular numpy array from the MaskedArray
		X_axis = np.array(results['param_' + param].data, dtype=float)

		for scorer, color in zip(sorted(scoring), ['g']):  # 'k', 'b', 'r']):
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
	# norm = StandardScaler().fit(data)

	# transform training data
	data_scaled = pd.DataFrame(norm.transform(data))
	data_scaled.columns = data.columns

	Y_scaled = data.OurRating
	X_scaled = data_scaled.drop(columns=['OurRating'])

	ratings_class = RatingsClassification(test_size=0.3)
	ratings_class.run_classification(X_scaled, Y_scaled)

	print(ratings_class.results.head(1000))

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
