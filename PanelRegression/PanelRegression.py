import pandas as pd
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
drive.mount("/content/gdrive")

#Read file with panel data 
data = pd.read_csv("PANEL DATA FILE PATH")

#HIV dictionary 
endg_var = "r_{ab}"
exog_var = ["HIV4 TONE", "log(SIZE)", "BTM", "TURNOVER", "PREFALPHA", "NASDAQ"]

pooled_y = data[endg_var]
pooled_x = data[exog_var]

pooled_x = sm.add_constant(pooled_x)

pooled_olsr_model_hiv4 = sm.OLS(endog=pooled_y, exog=pooled_x)
pooled_olsr_model_results_hiv4 = pooled_olsr_model_hiv4.fit()
print(pooled_olsr_model_results_hiv4.summary())


#LM dictionary

endg_var = "r_{ab}"
exog_var = ["LM TONE", "log(SIZE)", "BTM", "TURNOVER", "PREFALPHA", "NASDAQ"]

pooled_y = data[endg_var]
pooled_x = data[exog_var]

pooled_x = sm.add_constant(pooled_x)

pooled_olsr_model_lm = sm.OLS(endog=pooled_y, exog=pooled_x)
pooled_olsr_model_results_lm = pooled_olsr_model_lm.fit()
print(pooled_olsr_model_results_lm.summary())

#RF12

endg_var = "r_{ab}"
exog_var = ["r_{ab} RF12", "log(SIZE)", "BTM", "TURNOVER", "PREFALPHA", "NASDAQ"]

pooled_y = data[endg_var]
pooled_x = data[exog_var]

pooled_x = sm.add_constant(pooled_x)

pooled_olsr_model_rf12 = sm.OLS(endog=pooled_y, exog=pooled_x)
pooled_olsr_model_results_rf12 = pooled_olsr_model_rf12.fit()
print(pooled_olsr_model_results_rf12.summary())

#RF with FinBERT as encoder

endg_var = "r_{ab}"
exog_var = ["r_{ab} RFFin", "log(SIZE)", "BTM", "TURNOVER", "PREFALPHA", "NASDAQ"]

pooled_y = data[endg_var]
pooled_x = data[exog_var]

pooled_x = sm.add_constant(pooled_x)

pooled_olsr_model_rffin = sm.OLS(endog=pooled_y, exog=pooled_x)
pooled_olsr_model_results_rffin = pooled_olsr_model_rffin.fit()
print(pooled_olsr_model_results_rffin.summary())

#FinBERT + NN

endg_var = "r_{ab}"
exog_var = ["r_{ab} FinNN", "log(SIZE)", "BTM", "TURNOVER", "PREFALPHA", "NASDAQ"]

pooled_y = data[endg_var]
pooled_x = data[exog_var]

pooled_x = sm.add_constant(pooled_x)

pooled_olsr_model_finnn = sm.OLS(endog=pooled_y, exog=pooled_x)
pooled_olsr_model_results_finnn = pooled_olsr_model_finnn.fit()
print(pooled_olsr_model_results_finnn.summary())

