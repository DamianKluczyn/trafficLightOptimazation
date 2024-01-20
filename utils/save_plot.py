import matplotlib.pyplot as plt


# Saves a plot of total waiting times over epochs
def save_plot(total_waiting_times, model):
    plt.plot(list(range(len(total_waiting_times))), total_waiting_times)
    plt.xlabel("epochs")
    plt.ylabel("total time")
    plt.savefig(f'plots/time_vs_epoch_{model}.png')
    plt.show()


def save_evaluation_plot(avg_waiting_times_per_step, model):
    plt.plot(list(range(len(avg_waiting_times_per_step))), avg_waiting_times_per_step)
    plt.xlabel("step")
    plt.ylabel("average waiting time")
    plt.savefig(f'plots/evaluation_{model}.png')

    plt.close()


def save_base_plot(avg_waiting_times_per_step):
    plt.plot(list(range(len(avg_waiting_times_per_step))), avg_waiting_times_per_step)
    plt.xlabel("step")
    plt.ylabel("average waiting time")
    plt.savefig('../plots/evaluation_base.png')

    plt.close()
