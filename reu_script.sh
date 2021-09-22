#!/bin/bash

# run the program for each city of interest
python3 dRegression.py -c kdfw -m ridge  # run with the city kdfw with ridge regression
python3 dRegression.py -c kcys -m ridge  # run with the city kcys with ridge regression
python3 dRegression.py -c kroa -m ridge  # run with the city kroa with ridge regression

# python3 dRegression.py -c kdfw -m lasso  # run with the city kdfw with lasso regression
# python3 dRegression.py -c kcys -m lasso  # run with the city kcys with lasso regression
# python3 dRegression.py -c kroa -m lasso  # run with the city kroa with lasso regression

# python3 dRegression.py -c kdfw -m neural_net  # run with the city kdfw with a neural network
# python3 dRegression.py -c kcys -m neural_net  # run with the city kcys with a neural network
# python3 dRegression.py -c kroa -m neural_net  # run with the city kroa with a neural network

python3 create_plot.py          # create the plots for each city
python3 error_percentage_calculator.py  # get error overlap values