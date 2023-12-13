import matplotlib.pyplot as plt


# Saves a plot of total waiting times over epochs
def save_plot(total_waiting_times, model):
    plt.plot(list(range(len(total_waiting_times))), total_waiting_times)
    plt.xlabel("epochs")
    plt.ylabel("total time")
    plt.savefig(f'plots/time_vs_epoch_{model}.png')
    plt.show()
