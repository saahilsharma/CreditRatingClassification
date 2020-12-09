import numpy as np

class PCAHelper:
	@staticmethod
	def lower_bound(cv_results):

		best_score_idx = np.argmax(cv_results['mean_test_accuracy'])
		return (cv_results['mean_test_accuracy'][best_score_idx]
		        - cv_results['std_test_accuracy'][best_score_idx])

	@staticmethod
	def best_low_complexity(cv_results, param='dim_reduction__n_components'):
		threshold = PCAHelper.lower_bound(cv_results)
		candidate_idx = np.flatnonzero(cv_results['mean_test_accuracy'] >= threshold)
		best_idx = candidate_idx[cv_results['param_'+param][candidate_idx].argmin()]
		return best_idx