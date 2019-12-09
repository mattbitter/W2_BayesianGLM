# Matthew Bitter
# Bayes

# Libs
import pandas as pd
from pymc3 import *
import matplotlib.pyplot as plt
import numpy as np
import arviz as az


df = pd.read_csv("GLM_data_final.txt", delimiter='\t')

df['POS_AMT'] = pd.to_numeric(df['POS_AMT'])
df['POS_QTY'] = pd.to_numeric(df['POS_QTY'])

df['PPU'] = df['POS_AMT']/df['POS_QTY']



df['PPU_l'] = np.log(df['PPU'])
df['POS_QTY_SUM_l'] = np.log(df['POS_QTY'])

df_l = df[df['POS_QTY_SUM_l'] != 0]
df_l = df[df['PPU_l'] >= 0.04]

plt.subplot(2, 1, 1)
plt.scatter(df['PPU'], df['POS_QTY'], alpha=0.5)
plt.title('PPU and Quantity sold')
plt.xlabel('PPU')
plt.ylabel('Quantity Sold')

plt.subplot(2, 1, 2)
plt.scatter(df_l['PPU_l'], df_l['POS_QTY_SUM_l'], alpha=0.5)
plt.title('PPU and Quantity sold in log space')
plt.xlabel('Log of PPU')
plt.ylabel('Log of Quantity Sold')

plt.show()

# ### GLM

df_l.rename(columns={"PPU_l": "x"}, inplace=True)

with Model() as model:
    GLM.from_formula('POS_QTY_SUM_l ~ x', df_l[['POS_QTY_SUM_l', 'x']], family='normal')
    trace = sample(4000, cores=4)

var_min = df_l['x'].min()
var_max = df_l['x'].max()

plt.plot(df_l['x'], df_l['POS_QTY_SUM_l'], 'x')
plot_posterior_predictive_glm(trace, eval=np.linspace(var_min, var_max, 100))
plt.show()

traceplot(trace)
plt.show()

df_sum = summary(trace)

# # Bayes

occurrences = np.random.normal(2.65941, 0.06737, 500)

with Model() as model:

    prior_mu = Normal('prior_mu', mu=2.2, sigma=0.01)

    prior_sig = InverseGamma('prior_sig', alpha=3, beta=0.5)

    y_post = Normal('y_post', mu=prior_mu, sigma=prior_sig, observed=occurrences)

with model:
    # Sample from the posterior
    trace = sample(draws=3000, cores=2, tune=500, discard_tuned_samples=True)

traceplot(trace)
plt.show()

az.plot_joint(trace, kind='kde', fill_last=False)
plt.show()

plot_posterior(trace, round_to=3)
plt.show()

autocorrplot(trace)
plt.show()

pairplot(trace, kind='kde')
plt.show()
