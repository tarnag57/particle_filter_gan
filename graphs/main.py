import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g_df = pd.read_csv('../out/generator.csv')

series = np.stack([g_df.loc[i].to_numpy()[1:] for i in range(30)])
print(series)

average = np.log(np.mean(series, axis=0))
minimums = np.log(series.min(axis=0))
maximums = np.log(series.max(axis=0))
print(average.size)
print(minimums.size)
print(maximums.size)

x = np.arange(0, average.size)
plt.figure(figsize=(10, 6))
plt.plot(average, color="C0", label="Average Generator Loss")
plt.fill_between(x, maximums, minimums, color="C0", alpha=0.4)
plt.xlabel("Epochs")
plt.ylabel(r"$\ell(G_\theta)$")
plt.legend(loc=1, prop={'size': 12})
plt.show(block=True)
