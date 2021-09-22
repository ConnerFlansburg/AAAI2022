import pandas as pd
import pathlib as pth
import typing as typ
from formatting import printError, printWarn


# TODO: Use Average RMSE & Max RMSE

def calculate_increase(best: float, current: float):
    """ this calculates how much a value increases as a percentage """
    # ? This will produce negative values when the error decreases. Is that what we want?
    return round(((current/best)*100)-100, 3)


def averages(city: str):

    # *** Read in the Average Root Mean Squared Error *** #
    # Read in the values for Ridge Regression
    in_path = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'ridge' / f'rmse_avr.csv')
    ridge: pd.DataFrame = pd.read_csv(in_path, header=None, names=['Size', 'Ridge'])
    # Read in the values for Lasso Regression
    in_path = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'lasso' / f'rmse_avr.csv')
    lasso: pd.DataFrame = pd.read_csv(in_path, header=None, names=['Size', 'Lasso'])
    # Read in the values for Neural Network
    in_path = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'neural_net' / f'rmse_avr.csv')
    neural_net: pd.DataFrame = pd.read_csv(in_path, header=None, names=['Size', 'Neural Network'])
    # create a dataframe of the averages
    avr_df = pd.concat([ridge, lasso['Lasso'], neural_net['Neural Network']], axis=1, join='inner').set_index('Size')

    # *** Save the Dataframe *** #
    avr_df.to_csv(str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / f'combined_rsme_avr.csv'))

    # print(f'Average, Before Percentage Calculation:\n{avr_df}\n')  # ! debugging

    # *** Compute the Increase Percentages *** #
    # for each model, get the list of errors and loop over it
    for model in ['Ridge', 'Lasso', 'Neural Network']:

        # get the column of errors as a list
        error_values: typ.List[float] = list(avr_df[model])

        best_error: float = error_values[-1]  # get the best error

        increase = []  # this will hold the increase for every size

        # loop over the list of errors & compute the increase percentage
        for current_index, current in enumerate(error_values):

            if current_index == len(error_values):                              # if this is the last index
                increase.append(0)                                              # then the increase will be zero
            else:                                                               # if the increase is not zero, find it
                increase.append(calculate_increase(best=best_error, current=current))  # get the increase percentage

        # Now that we have a complete list of the increase for every size, replace the error col with it
        assert len(increase) == len(avr_df[model])  # test that the length is correct
        avr_df[model] = increase                    # replace the old column

    print(f'Percentage Calculation(Average):\n{avr_df}\n')

    # *** Save the Increase Percentages *** #
    avr_df.to_csv(str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / f'increases_rsme_avr.csv'))


def maximum(city: str):

    # *** Read in the Average Root Mean Squared Error *** #
    in_path = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'ridge' / f'rmse_max.csv')
    ridge: pd.DataFrame = pd.read_csv(in_path, header=None, names=['Size', 'Ridge'])

    in_path = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'lasso' / f'rmse_max.csv')
    lasso: pd.DataFrame = pd.read_csv(in_path, header=None, names=['Size', 'Lasso'])

    in_path = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'neural_net' / f'rmse_max.csv')
    neural_net: pd.DataFrame = pd.read_csv(in_path, header=None, names=['Size', 'Neural Network'])
    # create a dataframe of the maximums
    max_df = pd.concat([ridge, lasso['Lasso'], neural_net['Neural Network']], axis=1, join='inner').set_index('Size')

    # *** Save the Dataframe *** #
    max_df.to_csv(str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / f'combined_rsme_max.csv'))

    # print(f'Max, Before Percentage Calculation:\n{max_df}\n')  # ! debugging

    # *** Compute the Increase Percentages *** #
    # for each model, get the list of errors and loop over it
    for model in ['Ridge', 'Lasso', 'Neural Network']:

        # get the column of errors as a list
        error_values: typ.List[float] = list(max_df[model])

        best_error: float = error_values[-1]  # get the best error

        increase = []  # this will hold the increase for every size

        # loop over the list of errors & compute the increase percentage
        for current_index, current in enumerate(error_values):

            if current_index == len(error_values):                              # if this is the last index
                increase.append(0)                                              # then the increase will be zero
            else:                                                               # if the increase is not zero, find it
                increase.append(calculate_increase(best=best_error, current=current))  # get the increase percentage

            # ? The increase percentage is now calculated, what should I do with it?
        # Now that we have a complete list of the increase for every size, replace the error col with it
        assert len(increase) == len(max_df[model])  # test that the length is correct
        max_df[model] = increase                    # replace the old column

    print(f'Percentage Calculation (Max):\n{max_df}\n')

    # *** Save the Increase Percentages *** #
    max_df.to_csv(str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / f'increases_rsme_max.csv'))


if __name__ == "__main__":

    cities = ['kdfw', 'kcys', 'kroa']
    # cities = ['kdfw']

    for c in cities:
        averages(c)
        maximum(c)
