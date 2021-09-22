import pprint
import matplotlib.pyplot as plt
import pathlib as pth
import numpy as np
import pickle
import sys


def beaufort_scale(original: float) -> int:
    
    # speed should be in m/s, round it
    original = round(original, 1)

    # Determine the Beaufort Scale value & return it
    if original < 0.5:
        return 0

    elif 0.5 <= original <= 1.5:
        return 1

    elif 1.6 <= original <= 3.3:
        return 2

    elif 3.4 <= original <= 5.5:
        return 3

    elif 5.5 <= original <= 7.9:
        return 4

    elif 8 <= original <= 10.7:
        return 5

    elif 10.8 <= original <= 13.8:
        return 6

    elif 13.9 <= original <= 17.1:
        return 7

    elif 17.2 <= original <= 20.7:
        return 8

    elif 20.8 <= original <= 24.4:
        return 9

    elif 24.5 <= original <= 28.4:
        return 10

    elif 28.5 <= original <= 32.6:
        return 11

    elif 32.7 <= original:
        return 12

    # if this has been reached then an error has occured
    printWarn(f'ERROR: beaufort_scale expected flot got {original}, {type(original)}')
    sys.exit(-1)  # cannot recover from this error, so exit


def everyOverlap(city: str):
    
    # * Read in the Pickled Record * #
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'permutations_pickle.p')
    # this will be a list of lists, where each list is a single permutation
    permutations_by_smooth: np.array = np.array(pickle.load(open(jar, "rb")))

    # * For Each Smoothing Iteration, Get the Beaufort Scale Scores * #
    scores_by_smooth = []
    for wind_speeds in permutations_by_smooth:
        # get the b scale for a single permutation
        scores_by_smooth.append([beaufort_scale(i) for i in wind_speeds])

    # get the average of all the permutations
    avr = [beaufort_scale(i) for i in np.average(permutations_by_smooth, axis=0)]

    list_of_permutations = []
    for i in scores_by_smooth:       # loop over the outer list (i will be a permutation)
        print(i)  # ! debug ! #
        temp = []                    # this will hold the count value
        for j in range(1, 13):
            temp.append(i.count(j))  # get the number of times a b scale value occurs
        list_of_permutations.append(temp)  # add temp to final list

    # !!!!!!!!!!!! Debug !!!!!!!!!!!! #
    print('\n')
    pprint.pprint(list_of_permutations)
    print('\n')
    assert list_of_permutations[0] != []
    assert list_of_permutations[3] != []
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    avr_temp = []
    for j in range(1, 13):
        avr_temp.append(avr.count(j))  # get the number of times a b scale value occurs
    avr = avr_temp                     # replace the average with the new average
    pprint.pprint(avr)  # !!!! #

    # * Set Up the Plots * #
    fig, ax = plt.subplots()
    plt.title(f"{city} Beaufort Scale Overlap")
    plt.xlabel("Beaufort Scale Score")
    plt.ylabel("Probability")
    plt.rc('font', size=15)  # font size

    # * Plot the PDF * #
    for x in list_of_permutations:
        ax.plot(x)
    ax.plot(avr, linestyle='dashed', c='green')  # plot the average line
    ax.legend(loc='best', frameon=True)          # add the legend

    plt.show()

    fl = str(pth.Path.cwd() / 'output' / f'{city}' / 'overlap_plot.png')
    fig.tight_layout()  # tighten the layout
    plt.savefig(fl)  # save the plot
    return


def main():

    everyOverlap('kdfw')
    everyOverlap("kcys")
    everyOverlap("kroa")
    

def overlapPlots():
    
    main()


if __name__ == "__main__":
    main()
