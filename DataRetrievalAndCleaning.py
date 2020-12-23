import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import quandl


if __name__ == "__main__":
    not_junk_rating = ['A', 'A-', 'A *-', 'A- *', 'A- *-', 'A+', 'AA', 'AA-', 'AA- *-', 'AA+', 'AAA', 'BBB', 'BBB-',
                       'BBB- *-', 'BBB *+', 'BBB- *+', 'BBB+', 'BBB+ *-', 'BBB+ *+', 'BBBu', 'BBB-u']

    sp_1500 = pd.read_excel('S&P1500_Ratings.xlsx')
    sp_1500.dropna(subset=['RTG_SP_LT_LC_ISSUER_CREDIT'], inplace=True)
    sp_1500_clean = sp_1500[~sp_1500['RTG_SP_LT_LC_ISSUER_CREDIT'].isin(['NR', 'SD'])]
    sp_1500_clean['OurRating'] = sp_1500_clean['RTG_SP_LT_LC_ISSUER_CREDIT'].apply(
        lambda x: 1 if (x in not_junk_rating) else 0)

    sp_1500_clean['Ticker'] = sp_1500_clean['Ticker'].apply(lambda t: t.split(" ")[0].replace("/", "-"))

    quandl.ApiConfig.api_key = "ph9jSGcEBimkUQULCx4k"

    fundamentals = quandl.get_table('SHARADAR/SF1', dimension='MRY', ticker=sp_1500_clean['Ticker'].to_list())

    fund_max_date = fundamentals[['ticker','reportperiod']].groupby(['ticker']).max()

    latest_fundamentals = pd.merge(fundamentals, fund_max_date, on=['ticker','reportperiod'])

    fundamentals_subset = latest_fundamentals[['ticker','intexp','netinc','eps','ncfdebt','ncf','assets','cashneq',
                                               'receivables','payables','liabilities','equity','debt','ebit','ebitda',
                                               'marketcap','ev','roe','roa','fcf','roic','gp','opinc','grossmargin',
                                                'netmargin','evebitda','evebit','de','divyield','fcfps']
                                            ]

    result = pd.merge(sp_1500_clean, fundamentals_subset, left_on="Ticker", right_on="ticker",how="inner")
    result.drop(columns=['ticker']).to_csv("RatingsAndFundamentals.csv", header=True, index=False)

