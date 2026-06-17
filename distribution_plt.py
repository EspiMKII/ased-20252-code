import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle('Base Probability Distributions Used in Dataset', fontsize=16, fontweight='bold')

x_cont = np.linspace(0, 10, 500)
x_norm = np.linspace(-4, 4, 500)
x_geom = np.arange(1, 11)

# 1. Exponential
axes[0, 0].plot(x_cont, stats.expon.pdf(x_cont, scale=2), color='blue', lw=2)
axes[0, 0].fill_between(x_cont, stats.expon.pdf(x_cont, scale=2), color='blue', alpha=0.3)
axes[0, 0].set_title('Exponential Distribution')

# 2. Chi-Square (df=3)
axes[0, 1].plot(x_cont, stats.chi2.pdf(x_cont, df=3), color='orange', lw=2)
axes[0, 1].fill_between(x_cont, stats.chi2.pdf(x_cont, df=3), color='orange', alpha=0.3)
axes[0, 1].set_title('Chi-Square Distribution')

# 3. Normal
axes[1, 0].plot(x_norm, stats.norm.pdf(x_norm, loc=0, scale=1), color='green', lw=2)
axes[1, 0].fill_between(x_norm, stats.norm.pdf(x_norm, loc=0, scale=1), color='green', alpha=0.3)
axes[1, 0].set_title('Normal (Gaussian) Distribution')

# 4. Geometric (Discrete)
axes[1, 1].vlines(x_geom, 0, stats.geom.pmf(x_geom, p=0.4), colors='red', lw=4, alpha=0.6)
axes[1, 1].plot(x_geom, stats.geom.pmf(x_geom, p=0.4), 'ro')
axes[1, 1].set_title('Geometric Distribution')

plt.tight_layout()
plt.savefig('base_distributions.png', dpi=300)
plt.show()