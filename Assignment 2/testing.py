import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

N = 100
x = np.random.randn(N)
y = np.random.randn(N)
t = np.random.randint(0,2, N)
xx = np.random.randn(2)
yy = np.random.randn(2)

palette = sns.color_palette("bright", 2)

data = pd.DataFrame({'x' : x, 'y' : y, 't' : t})
# data2 = pd.DataFrame({'x' : xx, 'y' : yy, 't' : np.arange(2), 'size' : 100*np.ones(2)})
data3 = pd.DataFrame({'x' : np.append(x,xx), 'y' : np.append(y,yy), 't' : np.append(t, np.arange(2)), 'size' : np.append(10*np.ones(N), 100*np.ones(2))})

sns.scatterplot(data = data, x='x', y='y', hue = 't', style = 't', palette=palette)
# sns.scatterplot(data = data2, x='x', y='y', hue = 't', style = 't', size='size')
for i in range(2):  # for each cluster
    plt.scatter(xx[i], yy[i],
                color=palette[i], s=200, marker='X', edgecolor='k', linewidth=1.5)
plt.legend()
plt.show()