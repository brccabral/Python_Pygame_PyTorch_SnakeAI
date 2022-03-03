import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # interactive on


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())  # get current figure
    plt.clf()  # clear figure
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def plot_genetic(best_all_times, best_generation):
    display.clear_output(wait=True)
    display.display(plt.gcf())  # get current figure
    plt.clf()  # clear figure
    plt.title('Genetic...')
    plt.xlabel('Number of generations')
    plt.ylabel('Score')
    plt.plot(best_all_times)
    plt.plot(best_generation)
    plt.ylim(ymin=0)
    plt.text(len(best_all_times)-1,
             best_all_times[-1], str(best_all_times[-1]))
    plt.text(len(best_generation)-1,
             best_generation[-1], str(best_generation[-1]))
    plt.show(block=False)
    plt.pause(.01)
