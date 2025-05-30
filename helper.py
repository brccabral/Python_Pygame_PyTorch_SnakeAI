import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # interactive on


def plot(scores: list[int], mean_scores: list[float]):
    display.clear_output(wait=True)
    display.display(plt.gcf())  # get current figure
    plt.clf()  # clear figure
    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)
