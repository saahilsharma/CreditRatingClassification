# Data Manipulation
import numpy as np
import pandas as pd
import xlrd
import yahoo_fin.stock_info as si

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
import copy

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

def fillInNaColumnsFromYahooFinance(df) -> DataFrame:

       return df


if __name__ == "__main__":
       not_junk_rating = ['A','A-','A *-','A- *','A- *-','A+','AA','AA-','AA- *-','AA+','AAA','BBB','BBB-','BBB- *-',
       'BBB *+','BBB- *+','BBB+','BBB+ *-','BBB+ *+','BBBu','BBB-u']

       sp_1500 = pd.read_excel('S&P1500_Data.xlsx')
       sp_1500.drop(['Weight', 'Shares', 'Price','RTG_FITCH_SEN_UNSECURED',
              'RTG_MOODY_LONG_TERM', 'ESG_RATING', 'INT_COVERAGE_RATIO'], axis=1, inplace=True)

       sp_1500.dropna(subset=['RTG_SP_LT_LC_ISSUER_CREDIT'], inplace=True)
       sp_1500_clean = sp_1500[~sp_1500['RTG_SP_LT_LC_ISSUER_CREDIT'].isin(['NR','SD'])]
       sp_1500_clean['OurRating'] = sp_1500_clean['RTG_SP_LT_LC_ISSUER_CREDIT'].apply(lambda x: 'Not Junk' if(x in not_junk_rating) else 'Junk')

       #quote = si.get_quote_table(sp_1500_clean['Ticker'][0])
       fin = si.get_financials("PFG")
       print(fin)

# plt.scatter(sp_1500_clean['CF_FREE_CASH_FLOW'], sp_1500_clean['OurRating'], c='g', s=4)
# plt.show()
# plt.scatter(sp_1500_clean['BEST_EPS'], sp_1500_clean['OurRating'], c='g', s=4)
# plt.show()
# plt.scatter(sp_1500_clean['RETURN_ON_ASSET'], sp_1500_clean['OurRating'], c='g', s=4)
# plt.show()
# plt.scatter(sp_1500_clean['EBIT'], sp_1500_clean['OurRating'], c='g', s=4)
# plt.show()

#TODO: Plot a few more scatter plots to see the relationships
#TODO: Decide on approach for classification
#   1). Logistic Regression vs LDA vs QDA vs Non-linear (Splines)
#   2). Parameter selection
#   3). Implement Methods using Cross Validation
#   4). Plot Results of MSE

