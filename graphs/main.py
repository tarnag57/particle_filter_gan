import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../src')
import data

def loss_graph():
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


def get_counts(samples):
    # Remove $ and \n
    norm_samples = [s.translate({ord(i): None for i in '$\n'})
                    for s in samples]
    counts = np.zeros(10)
    for sample in norm_samples:
        counts[len(sample)] += 1

    return counts


def length_graph():

    filename = '../out/short_samples.txt'
    f = open(filename, 'r')
    samples = f.readlines()
    gen_counts = get_counts(samples)[1:]

    names = data.prepare_names()[:200]
    real_counts = get_counts(names)[1:]

    print(gen_counts)
    print(real_counts)

    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, gen_counts, width, label='Generated Names')
    rects2 = ax.bar(x + width/2, real_counts, width, label='Real Names')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Length of generated and real names')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()

    plt.show()


length_graph()
