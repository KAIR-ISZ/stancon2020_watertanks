import sys

from lib import stan_utility
import pystan
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from lib.DA_tools import ribbon_plot, get_quantiles
from lib.DA_colors import DARK
from scipy.interpolate import BSpline
from datetime import datetime
import pickle


plt.style.context('seaborn-white')
mpl.rcParams['figure.dpi'] = 200

# number of trajectories
N = 20 
# number of samples per trajectory
M=1000


wide_h3 = pd.read_csv('trajectories_scen0.csv').iloc[:, 3::3]

y_med = np.median(wide_h3.values.T,axis=0)
print(y_med.shape)
centered = wide_h3-np.expand_dims(y_med,axis=1)
centered['time']=[*range(0,M)]

# number of trajectories
trajectories0 = pd.wide_to_long(centered, stubnames=['h3'], i='time', j='experiment', suffix='_\d+')
trajectories0.index = trajectories0.index.droplevel(1)
trajectories1 = trajectories0.reset_index().head(N*1000)


# Z tego moznaby zrobic funckje

spl_order = 3
num_knots = 7
knot_list = np.quantile(trajectories1.time, np.linspace(0, 1, num_knots))

knots = np.pad(knot_list, (spl_order, spl_order), mode="edge")
# Design matrix

B = BSpline(knots, np.identity(num_knots + 2),
            k=spl_order)(trajectories1.time.values[0:1000])



spline_fit2_ppc = stan_utility.compile_model('spline_fit2_ppc.stan')
data_fit2 = dict(N=B.shape[0],
                 K=B.shape[1],
                 L=1000,
                 x=B,
                 y=trajectories1.h3.values)
# fitowanie
ppc = spline_fit2_ppc.sampling(data=data_fit2, iter=1000, warmup=0, 
                           chains=1, 
                           refresh=1000,
                           algorithm='Fixed_param',
                           seed=6062020)

y_pred = ppc.extract()['y_pred']
beta = ppc.extract()['beta']
mn_beta = np.mean(beta, axis=0)
sd_beta = np.std(beta, axis=0)
spline_components = B@np.diag(mn_beta)
y_m = np.max(spline_components, axis=0)
x_m = np.argmax(spline_components, axis=0)

# dane oryginalne do wizualizacji
# wide_h3 = pd.read_csv('trajectories_scen0.csv').iloc[:, 3::3]

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

ax2 = axes[0]
#ax2.set_ylim((-1, 14))

for i in range(num_knots+2):
    ax2.plot(trajectories1.time.values[0:1000],
             (spline_components[:, i]), color=DARK, zorder=0)
ax2.errorbar(x_m, y_m, ls='none', yerr=2 *
             sd_beta[i]*y_m[i], color='black', capsize=4)
ax2.text(0, 6, s='B-spline functions', color=DARK)
# ax2.annotate(xy=(x_m[7], y_m[7]), s='95% confidence interval',
#              xytext=(500, 10), arrowprops={'arrowstyle': '->'})
ax2.set_ylabel('B-spline values')

# ax2.set_yticks([0, 14])
# ax2.set_yticklabels(['0 cm', '14 cm'])
ax3 = axes[1]
ax3 = ribbon_plot(trajectories1.time.values[0:1000], y_pred, ax3, probs=[
                  2.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97.5])
# qs = get_quantiles(centered.values.T, [2.5, 50, 97.5])
# ax3.plot(trajectories1.time.values[0:1000],
#          qs[0, :], color='black', linestyle='--')
# ax3.plot(trajectories1.time.values[0:1000], qs[1, :], color='black')
# ax3.plot(trajectories1.time.values[0:1000],
#          qs[2, :], color='black', linestyle='--')
ax3.set_ylabel('Water level')
ax3.set_xlabel('Time')
# ax3.annotate(xy=(600, qs[0, 600]), s='95% measurements interval', xytext=(
#     400, 4), arrowprops={'arrowstyle': '->'})
# ax3.annotate(xy=(400, 10),
#              s='95% predictive interval',
#              xytext=(0, 12),
#              arrowprops={'arrowstyle': '->', 'color': DARK},
#              color=DARK)
# ax3.set_yticks([0, 14])
# ax3.set_yticklabels(['0 cm', '14 cm'])
ax3.set_xticks([0, 500, 1000])
ax3.set_xticklabels(['0 s', '500 s', '1000 s'])
fig.tight_layout()
fig.savefig('Spline_approx_healthy_ppc.png')

plt.show()
