import typing

import matplotlib.pyplot as plt
import typing as typ
import pandas as pd
import pathlib as pth
import numpy as np
import pickle
import sys
from formatting import printError, printWarn, printSuccess

SIZES: typ.Tuple = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
                    1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2514)

WORST_PERMS: typ.List[int] = []
AVR_PERM = []

PRINT_OVERLAP: bool = False  # should info about the overlap be printed

MODEL: str = 'ridge'

# *** THIS IS GOOD TO GO *** #
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

    # if this has been reached then an error has occurred
    printWarn(f'ERROR: beaufort_scale expected flot got {original}, {type(original)}')
    sys.exit(-1)  # cannot recover from this error, so exit


# *** THIS IS GOOD TO GO *** #
def get_testing(city: str):
    """ get the beaufort scale scores of the test set """

    # * Read in the File * #
    fl = str(pth.Path.cwd() / 'data' / f'{city}_processed_data.csv')
    df: pd.DataFrame = pd.read_csv(
        fl, dtype=float, na_values='********',
        parse_dates=['date'], index_col=0
    )
    df.sort_values(by='date')  # sort the dataframe by date

    # !!!!!!!!!! Get the NaN Report !!!!!!!!!! #
    # zero_report(df, csv)
    # !! Create a Row in the Metadata Table !! #
    # row = [len(df.index), len(df.columns)]  # (stem, instances, features)
    # metadata.append(row)
    # !!! Create a Row in the Score Report !!! #
    # row = scale_values_report(df, csv)
    # scores.append(row)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # * Get rid of NaN & infinity * #
    df = df.replace('********', np.nan).replace(np.inf, np.nan)
    df = df.dropna(how='any', axis=1)

    # * Get the Wind Speeds Values from the Data * #
    data = list(df['OBS_sknt_max'])

    # * Slice the Data into Training & Testing * #
    train_data = data[: int(len(data) * .80) + 1]  # up to & including 80%

    # * convert wind speed to b scale * #
    return [beaufort_scale(i) for i in train_data]


# *** THIS IS GOOD TO GO *** #
def test_set_values(city: str):
    """ get the b scores of the test set """

    # * read in the test set & convert the wind speeds to b scale * #
    testing = get_testing(city)

    # * get the number of times that each b scale value occurs as a % * #
    count_arr = np.bincount(testing, minlength=13)  # get the number of times each value occurs

    test_count = []
    for i in count_arr:
        # get the number of times a b scale value occurs as a percentage
        test_count.append(round(i/len(testing), 3))

    return test_count


# *** THIS IS GOOD TO GO *** #
def new_set_values(training, cumulative: bool = False):
    """ get the b scores of the "newest" 100 """

    # transform the input values into b scores
    for index, value in enumerate(training):
        training[index] = [beaufort_scale(j) for j in value]

    if cumulative:
        # this should create the cumulative list
        for i in range(1, len(training)):
            training[i] = training[i] + training[i - 1]

        try:  # !!!!! Check the Created Cumulative List !!!!! #
            assert len(training[0]) == 100
            # printWarn('Assert len(training[1]) == 100  PASSED!')
            assert len(training[1]) == 200
            # printWarn('Assert len(training[2]) == 200  PASSED!')
            assert len(training[2]) == 300
            # printWarn('Assert len(training[3]) == 300  PASSED!')
        except AssertionError as e:
            printError(str(e))  # print the error
            printWarn(f'Length of 1st index is {len(training[0])}')
            printWarn(f'1st index is: {training[0]}')
            sys.exit(-1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # get the number of times each value occurs as an integer
    training = [list(np.bincount(j, minlength=13)) for j in training]
    # print(f'The Results of the Bin Count:\n{pprint.pprint(training)}\n')  # ! for debugging
    training = [[round(ii/np.sum(i), 3) for ii in i] for i in training]  # take the int & make it a percentage
    # print(f'Percentage Conversion Results:\n{pprint.pprint(training)}\n')  # ! for debugging
    # print(np.array(training))

    try:
        assert len(training) == 25
    except AssertionError:
        printError(f'Length of Training is {len(training)}')
        printError(f'Cumulative is {cumulative}')
        sys.exit(-1)
    for t in training:
        assert len(t) == 13
    # print(f'Training:\n{training}\n')

    return training


# *** THIS IS GOOD TO GO *** #
def find_worst_permutation(city: str, size: int):

    # * Read in the Pickled Record * #
    # spikes[Permutation Number][Error Score]
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / f'{MODEL}' / 'spike_record.p')
    spikes: typ.Dict[int, typ.Dict[int, float]] = pickle.load(open(jar, "rb"))
    # printWarn('Spike Dictionary')  # ! for debugging
    # printWarn('P#: Error')         # ! for debugging
    # pprint.pprint(spikes[100])     # ! for debugging

    # * Read in the Pickled Training Sets * #
    # training_sets[Size][Permutation][Wind Speed]
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / f'{MODEL}' / 'train_set_record.p')
    training_sets: typ.Dict[int, typ.Dict] = pickle.load(open(jar, "rb"))
    # printWarn('Training_sets')             # ! for debugging
    # printWarn('Size: P#: Err')             # ! for debugging
    # pprint.pprint(training_sets, depth=2)  # ! for debugging

    # * For the Passed Size, Find the Worst Permutation * #
    # spikes[size] will contain a dictionary that holds the error score values
    # for each smooth iteration, and the inner dict is keyed by the smooth_iter
    # (which is equal to permutation value). So find the key with the worst
    # error score.

    # *** Get Permutation 1 *** #
    # get a list that takes every value of size (for a p1) & adds it in order
    # printWarn('Before P1:')                        # ! for debugging
    # pprint.pprint(spikes[size], sort_dicts=False)  # ! for debugging

    p1 = max(spikes[size], key=spikes[size].get)  # get the perms index
    permutation1 = []
    for outer in training_sets.keys():
        permutation1.append(training_sets[outer][p1])
    del spikes[size][p1]                          # remove 1st highest

    # printWarn(f'Removed Key {p1}')                 # ! for debugging
    # printWarn('Before P2:')                        # ! for debugging
    # pprint.pprint(spikes[size], sort_dicts=False)  # ! for debugging

    # *** Get Permutation 2 *** #
    p2 = max(spikes[size], key=spikes[size].get)  # get the perms index
    permutation2 = []
    for outer in training_sets.keys():
        permutation2.append(training_sets[outer][p2])
    del spikes[size][p2]                          # remove 2nd highest

    # printWarn(f'Removed Key {p2}')                 # ! for debugging
    # printWarn('Before P3:')                        # ! for debugging
    # pprint.pprint(spikes[size], sort_dicts=False)  # ! for debugging

    # *** Get Permutation 3 *** #
    p3 = max(spikes[size], key=spikes[size].get)  # get the perms index
    permutation3 = []
    for outer in training_sets.keys():
        permutation3.append(training_sets[outer][p3])
    # (no need to del 3 as we are now done with the read in pickles)

    # printWarn(f'Selected Key {p1}')                # ! for debugging

    assert p1 != p2      # Check that the keys are unique
    assert p1 != p3      # (i.e. that we didn't choose the
    assert p2 != p3      # same permutation more than once)
    assert p1 is not p2
    assert p1 is not p3
    assert p2 is not p3

    # this will allow the plot to label them by perm number later on
    global WORST_PERMS
    WORST_PERMS = [p1, p2, p3]
    # print(f'Perms Selected: {WORST_PERMS}')      # ! for debugging

    # print(f'\nThe Permutations found by find_worst_permutation are:'  # ! For Debugging
    #       f'\n{permutation1}\n{permutation2}\n{permutation3}\n')      # ! For Debugging

    return permutation1, permutation2, permutation3


# *** THIS IS GOOD TO GO *** #
def get_values(city: str, spike_location: typing.Optional[int] = None, cumulative: bool = False):
    """ This gets the list of permutations that we want to plot."""

    # * Get the List of Permutations to be Plotted * #
    if spike_location:  # if we are only plotting the top 3
        p1, p2, p3 = find_worst_permutation(city, spike_location)
        permutations = [p1, p2, p3]

        # print(f'\nThe list created by get_values is:\n{permutations}')  # ! for debugging
        # printWarn(f'\nThe length of permutation 1 is {len(p1)}')        # ! for debugging

    else:  # otherwise grab all the permutations
        # read in the pickled record
        jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / f'{MODEL}' / 'perm2.p')
        # this will be a list of lists, where each list is a single permutation
        permutations = pickle.load(open(jar, "rb"))

    # print(f'Shape of Overall: {overall.shape}\n{overall}\n')  # ! for debugging
    # print(f'Length of the Average Permutation Created: {len(average_b_score)}')  # ! for debugging
    # print(f'{average_b_score}\n')  # ! for debugging

    # new_val is a list of of permutations,
    # where each permutation is a list of buckets,
    # and where each bucket is a lit of how often each b score appears in the newest 100 instances
    new_val = []
    # for each smoothening permutation remover
    for i in permutations:
        # for each permutation, convert it to the b scale &
        # get the frequency of each b score as a percentage
        new_val.append(new_set_values(i, cumulative))

    # print(f"the size of permutations is {len(new_val)}")                # ! for debugging
    # print(f"the inner list size of permutations is {len(new_val[0])}")  # ! for debugging

    return new_val


# *** THIS IS GOOD TO GO *** #
def compute_overlap(city: str, spike_location: typing.Optional[int] = None):
    """
    This computes the overlap between the test set & the various training sets.
    It will return permutations which is a list of 10 permutations. Each of the
    10 permutations is a list of 25 buckets, and each bucket is a single value
    representing how the newest 100 overlap with the test set.

    NOTE: spike location should be a size value (100, 200, etc.) NOT an index of that size.

    """

    # !!! BUG: This is only returning 3 Permutations even when we want 10

    # get the frequency (%) of b scale values from the test set
    test = test_set_values(city)
    # get the list of new 100's in a list of lists with the outer list being the permutations
    permutations = get_values(city, spike_location)

    if spike_location:    # if we are plotting a spike,
        outer_range = 3   # we should only have the 3 worst permutations
    else:                 # if we are not plotting a spike,
        outer_range = 10  # then we are plotting all 10 permutations

    # check that the assumed size of the outer list is correct
    assert len(permutations) == outer_range

    for i in range(outer_range):                         # loop over each permutation
        assert len(permutations[i]) == 25                # we should now be dealing with the buckets inside a perm
        for j in range(25):                              # loop over every bucket in the permutation
            assert len(permutations[i][j]) == 13         # make sure that the bucket is storing b score info
            assert len(permutations[i][j]) == len(test)  # make sure that it's length is the same as the test set
            # get the bitwise min of the b score frequencies in the permutation & the test set (the overlap)
            # & then sum the mins to get the total overlap
            permutations[i][j] = np.sum(np.minimum(permutations[i][j], test))

    # permutations is now a list of 10 (or 3 if cumulative) permutations.
    # Each of the permutations is a list of 25 buckets.
    # Each bucket is a single value representing how either
    # the newest 100 overlap with the test set or
    # how total test set thus far overlaps with the test set (in the cumulative case)
    return permutations


# *** THIS IS GOOD TO GO *** #
def compute_cumulative_overlap(city: str, spike_location: typing.Optional[int] = None):
    """
    This computes the overlap between the test set & the various training sets.
    It will return permutations which is a list of 10 permutations. Each of the
    10 permutations is a list of 25 buckets, and each bucket is a single value
    representing how the newest 100 overlap with the test set.

    NOTE: spike location should be a size value (100, 200, etc.) NOT an index of that size.

    """

    # get the frequency (%) of b scale values from the test set
    test = test_set_values(city)
    # get the list of new 100's in a list of lists with the outer list being the permutations
    permutations: typ.List[typ.List[typ.List[float]]] = get_values(city, spike_location, cumulative=True)

    if spike_location:
        outer_range = 3
    else:
        outer_range = 10

    assert outer_range == len(permutations)
    # print(f'Permutations outer length: {len(permutations)}')     # ! for debugging
    # print(f'Permutations inner length: {len(permutations[0])}')  # ! for debugging

    # print(f'Permutations (as gotten by compute_cumulative_overlap):\n'  # ! for debugging
    #      f'{[[print(ii) for ii in i] for i in permutations]}\n')       # ! for debugging

    # frames = []  # ! for debug
    for i in range(outer_range):                         # loop over each permutation
        if PRINT_OVERLAP:                                # *** print info if requested *** #
            print(f"\n\033[33m{f'Permutation {i}'}\033[00m")
        assert len(permutations[i]) == 25                # check that we have the correct number of buckets
        for j in range(25):                              # loop over every bucket in the permutation
            assert len(permutations[i][j]) == 13         # check that we are looking at b score info
            assert len(permutations[i][j]) == len(test)  # check that it's the same size as the test data

            # !!!!!!!!!!!!!!!!!!!! Create a Dataframe for Debugging Purposes !!!!!!!!!!!!!!!!!!!! #
            # df = pd.DataFrame(columns=range(1, 14), index=['Perm', 'Test', 'Min'])
            # df.loc['Perm'] = permutations[i][j]
            # df.loc['Test'] = test
            # df.loc['Min'] = np.minimum(permutations[i][j], test)
            # df.loc['Avr'] = [df[1].mean(), df[2].mean(), df[3].mean(), df[4].mean(), df[5].mean(), df[6].mean(),
            #                  df[7].mean(), df[8].mean(), df[9].mean(), df[10].mean(), df[11].mean(), df[12].mean(),
            #                  df[13].mean()]
            # df.loc['i'] = i
            # df.loc['j'] = j
            # frames.append(df.transpose())  # add the dataframe to the list of dataframes
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # * I am adding the sum because I want to total overlap that this permutation has with the test set.
            # *     What this SHOULD do is get a list of minimum overlap for each b score,
            # *     & then sum it to get the total overlap for this permutation
            # get the bitwise min of the b scores in the permutation & the test set (the overlap)
            # & then get the sum (the total overlap)
            partial_overlaps: np.ndarray = np.minimum(permutations[i][j], test)
            total_overlap: float = partial_overlaps.sum()
            permutations[i][j] = total_overlap

            if j < 9 and PRINT_OVERLAP:  # *** print info if requested *** #
                print(f'Size  {SIZES[j]}, Overlap {round(total_overlap*100, 2)}%')
            elif PRINT_OVERLAP:
                print(f'Size {SIZES[j]}, Overlap {round(total_overlap*100, 2)}%')

    # frame = pd.concat(frames, axis=0)  # create 1 dataframe from all the dataframes    # ! for debug
    # frame.to_csv(str(pth.Path.cwd() / 'logs' / f'cumulative_report.csv'), index=True)  # ! for debug
    permutation_overlaps: typ.List[typ.List[typ.List[float]]] = permutations

    # ********* Print Info about Permutations ********* #
    if PRINT_OVERLAP:
        print()
        for index, val in enumerate(permutation_overlaps):
            sys.stdout.write(f"Permutation {index} [")
            for ii in val:
                sys.stdout.write(f"{ii:1.3f}, ")
            sys.stdout.write(f"]\n")
        sys.stdout.write(f"\n")
        sys.stdout.flush()
        for i, v in enumerate(permutation_overlaps):
            sys.stdout.write(f"Permutation {i}: Min = {min(v):1.3f}, Max = {max(v):1.3f}, Mean = {np.mean(v):1.3f}\n")
    # ************************************************* #

    # permutations is now a list of 10 permutations.
    # Each of the 10 permutations is a list of 25 buckets.
    # Each bucket is a single value representing how the newest 100 overlap with the test set
    return permutation_overlaps


def get_total_average(city: str, cumulative: bool = False):

    if cumulative:
        # get the cumulative over lap for every permutation
        overlap_all_permutations = compute_cumulative_overlap(city)
    else:
        # This will get a list of lists. The outer list will be of length 10 & contain an
        # entry for each permutation. The inner list will be of length 25 & contain the
        # overlap for each bucket.
        overlap_all_permutations = compute_overlap(city)
    # now take an average of the overlap
    overlap_avr_permutation = np.average(overlap_all_permutations, axis=0)

    return overlap_avr_permutation


def plot_overlap(city: str, spike_location: typing.Optional[int] = None, cumulative: bool = False):
    """ Plot the overlap for a single city """

    # * Set Up the Plots * #
    if cumulative:
        plt.title(f"{city} Beaufort Scale Overlap: Training Set (cumulative) vs Testing")
    else:
        plt.title(f"{city} Beaufort Scale Overlap: Newest 100 Values vs Testing")
    ax = plt.gca()                  # get the current ax
    ax.set_ylabel('Overlap')        # label the y-axis
    if cumulative:
        ax.set_xlabel('Size')       # label the x-axis
    else:
        ax.set_xlabel('Breakpoint')  # label the x-axis
    ax.set_xlim([1, 25])             # set the range of the x ticks
    fig = plt.gcf()                  # get the current figure
    fig.set_size_inches(14, 9)       # set the size of the image (width, height)
    plt.rc('font', size=15)          # font size

    # how many degrees the x-axis labels should be rotated
    # rotate = 0
    rotate = 45
    # rotate = 90

    # *********************************** Create a Dataframe *********************************** #

    # *** Get the List of Overlap Values & the Averages *** #
    if spike_location:  # we have a spike to plot around

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Error with Overlap Calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # cumulative_permutations = compute_cumulative_overlap(city)  # get the cumulative overlap
        # first_100_permutations = compute_overlap(city)  # get the non-cumulative overlap
        #
        # assert len(cumulative_permutations) == len(first_100_permutations)  # they should have the same length
        #
        # print(f'For {city}:')
        # for p in range(len(cumulative_permutations)):
        #     c_value = cumulative_permutations[p][0]
        #     f100_value = first_100_permutations[p][0]
        #     if c_value == f100_value:
        #         printSuccess(f'Perm {p}, 1st Value is {c_value}')
        #     else:
        #         printError(f'Perm {p}, c_value ({c_value}) != f100_value({f100_value})')
        #
        # sys.exit(-1)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

        if cumulative:  # we want to plot the spike cumulatively
            permutations = compute_cumulative_overlap(city, spike_location)
            permutations.append(np.average(permutations, axis=0))          # add the "local" average to the list
            permutations.append(get_total_average(city, cumulative=True))  # add the total average to the list

        else:           # we want to plot only the newest 100
            permutations = compute_overlap(city, spike_location)
            permutations.append(np.average(permutations, axis=0))  # add the "local" average to the list
            permutations.append(get_total_average(city))           # add the total average to the list

        # printWarn(f'Permutations Shape from Spike Location: {np.shape(permutations)}')  # ! for debugging

        cols = [f'Permutation {WORST_PERMS[0]}', f'Permutation {WORST_PERMS[1]}', f'Permutation {WORST_PERMS[2]}']
        perms = [f'Permutation {WORST_PERMS[0]}', f'Permutation {WORST_PERMS[1]}', f'Permutation {WORST_PERMS[2]}']
        cols.append('Local Average')
        cols.append('Total Average')

    else:               # we don't have a spike

        if cumulative:  # we want to plot all the permutations cumulatively
            permutations = compute_cumulative_overlap(city)
        else:           # we want to plot only the newest 100 of ever permutation
            permutations = compute_overlap(city)

        # printWarn(f'Permutations Shape from Non-Spike Location: {np.shape(permutations)}')  # ! for debugging

        permutations.append(np.average(permutations, axis=0))  # here total avr = local avr so only add it once

        cols = ['Permutation 1', 'Permutation 2', 'Permutation 3', 'Permutation 4', 'Permutation 5',
                'Permutation 6', 'Permutation 7', 'Permutation 8', 'Permutation 9', 'Permutation 10']
        perms = ['Permutation 1', 'Permutation 2', 'Permutation 3', 'Permutation 4', 'Permutation 5',
                 'Permutation 6', 'Permutation 7', 'Permutation 8', 'Permutation 9', 'Permutation 10']
        cols.append('Total Average')

    # *** Transpose the List of Permutations *** #
    permutations = np.transpose(permutations)

    try:
        permutations_df = pd.DataFrame(permutations, columns=cols)
    except ValueError as e:
        # lineNm = sys.exc_info()[-1].tb_lineno          # get the line number of error
        # printError(''.join(traceback.format_stack()))  # print stack trace
        printError(f'{e}')  # print the error message
        if cumulative:
            printError('Cumulative is True')
        else:
            printError('Cumulative is False')
        printWarn(f'Cols len is {len(cols)}, permutations len is {permutations.shape}')
        printWarn(f'cols: {cols}')
        printWarn(f'permutations: {permutations}')
        sys.exit(-1)

    # ****************************************************************************************** #

    for p in perms:
        permutations_df[p].plot(ax=ax,
                                kind='line',
                                x_compat=True,
                                rot=rotate,      # how many degrees to rotate the x-axis labels
                                use_index=True,
                                grid=True,
                                legend=True,
                                # marker='o',    # what type of data markers to use?
                                # mfc='black'    # what color should they be?
                                )

    g = permutations_df['Local Average'].plot(ax=ax,
                                              kind='line',
                                              x_compat=True,
                                              rot=rotate,      # how many degrees to rotate the x-axis labels
                                              use_index=True,
                                              grid=True,
                                              legend=True,
                                              color='black',
                                              style='-.',
                                              # marker='o',    # what type of data markers to use?
                                              # mfc='black'    # what color should they be?
                                              )
    if spike_location:
        g = permutations_df['Total Average'].plot(ax=ax,
                                                  kind='line',
                                                  x_compat=True,
                                                  rot=rotate,      # how many degrees to rotate the x-axis labels
                                                  use_index=True,
                                                  grid=True,
                                                  legend=True,
                                                  color='red',
                                                  style='--',
                                                  # marker='o',    # what type of data markers to use?
                                                  # mfc='black'    # what color should they be?
                                                  )

    g.set_xticks(range(len(SIZES)))

    # if plotting cumulative score, set labels to show the total size at that point
    if cumulative:
        g.set_xticklabels(SIZES)

    # * Save the Plot * #
    if spike_location and cumulative:
        stem: str = f'{city}_cumulative_overlap_{spike_location}.png'
        fl = str(pth.Path.cwd() / 'output' / f'{city}' / 'overlap_plots' / stem)
    elif spike_location:
        fl = str(pth.Path.cwd() / 'output' / f'{city}' / 'overlap_plots' / f'{city}_overlap_{spike_location}.png')
    else:
        fl = str(pth.Path.cwd() / 'output' / f'{city}' / 'overlap_plots' / f'{city}_overlap_.png')
    fig.tight_layout()  # tighten the layout
    plt.savefig(fl)     # save the plot
    # plt.show()
    plt.clf()           # clear the plot for the next city
    return


def main(model):

    # set what model we are looking at
    global MODEL
    MODEL = model

    # * Compute & Plot the Overlap with the Test Set * #
    plot_overlap('kdfw', 500, False)   # compute the non-cumulative overlap
    plot_overlap('kdfw', 500, True)    # compute the cumulative overlap
    printSuccess(' * kdfw Completed   \u2713')

    plot_overlap('kcys', 500, False)   # compute the non-cumulative overlap
    plot_overlap('kcys', 500, True)    # compute the cumulative overlap
    printSuccess(' * kcys 1 Completed \u2713')
    plot_overlap('kcys', 1100, False)  # compute the non-cumulative overlap
    plot_overlap('kcys', 1100, True)   # compute the cumulative overlap
    printSuccess(' * kcys 2 Completed \u2713')

    plot_overlap('kroa', 500, False)   # compute the non-cumulative overlap
    plot_overlap('kroa', 500, True)    # compute the cumulative overlap
    printSuccess(' * kroa 1 Completed \u2713')
    plot_overlap('kroa', 1200, False)  # compute the non-cumulative overlap
    plot_overlap('kroa', 1200, True)   # compute the cumulative overlap
    printSuccess(' * kroa 2 Completed \u2713')


if __name__ == "__main__":
    main()
