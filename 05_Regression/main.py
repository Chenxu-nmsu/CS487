# import libraries
import mylasso, mylr, myransac, myridge, myrf, mynormal, logistic_regression
import argparse
from time import process_time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys

##################
### ---> pre-process CRP dataset
# df = pd.read_csv('CRP.csv', header=0, low_memory=False)
# df=df.fillna(0)
#
# # Get the total Bio hourly production ("BIOGAS" + "BIOMASS")
# df = df[["BIOGAS", "BIOMASS"]]
# df["BIO_TOTAL"] = df["BIOGAS"] + df["BIOMASS"]
#
# # Remove everything and use only the "BIO_TOTAL"
# df_bio = df["BIO_TOTAL"]
#
# # Sum every 24 rows (this will get me the sum for every day)
# df_daily = df_bio.groupby(df_bio.index // 24).sum()
#
# # Traverse the DF with step of 7 days
# num_rows = df_daily.shape[0]
#
# weekly = []
# prev_i = 0
# num_days = 7
# for i in range(num_days, num_rows, num_days):
#     week = []
#     print("week", i/num_days)
#     first = i - num_days
#     print("First day of week:", first)
#     last = i
#     print("Last day of week:",last)
#     for day in range(first, last):
#         print("day:", day)
#         week.append(df_daily.iloc[day])
#     weekly.append(week)
#
# # I have every week as a list, and I will use this to create a new DF
# weekly_columns = ["1", "2", "3", "4", "5", "6", "7"]
# weekly_df = pd.DataFrame(weekly, columns=weekly_columns)
#
# weekly_df['Target'] = weekly_df.mean(axis=1).shift(-1).fillna(0)
#
# weekly_df.to_csv("CRP_new.csv", index=None)
###################

if __name__ == '__main__':

    # add arguments to input specific classifier and data to execute certain group of codes.
    # arguments of parameter in each classifier are also added.
    parser = argparse.ArgumentParser()
    parser.add_argument('regression_approach', help='regression_approach', choices=["lr", "ransac", "ridge", 'lasso', 'rf', 'normal', 'logistic'])
    parser.add_argument('data', help='Dataset', choices=['housing.txt', 'CRP_new.csv'])

    # parameters in lr
    # None

    # parameters in ransac
    parser.add_argument('--min_sample', help='min_sample in [ransac]', type=int)
    parser.add_argument('--max_trials', help='max_trials in [ransac]', type=int)
    parser.add_argument('--loss', help='loss in [ransac]', type=int)
    parser.add_argument('--residual_threshold', help='residual_threshold in [ransac]', type=int)

    # parameters in ridge
    parser.add_argument('--alpha1', help='alpha in [ridge]', type=float)
    parser.add_argument('--solver', help='solver in [ridge]', type=str)

    # parameters in lasso
    parser.add_argument('--alpha2', help='alpha in [lasso]', type=float)

    # parameters in rf[nonlinear]
    parser.add_argument('--n_estimators', help='alpha in [rf]', type=int)
    parser.add_argument('--criterion', help='alpha in [rf]', type=str)
    parser.add_argument('--n_jobs', help='alpha in [rf]', type=int)

    # parameters in logistic regression
    # None

    args = parser.parse_args()
    args = vars(parser.parse_args())

    if args['data'] == "housing.txt":
        df = pd.read_csv(args['data'], header=None, sep='\s+')
        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

    elif args['data'] == "CRP_new.csv":
        df = pd.read_csv('CRP_new.csv', header=0, low_memory=False)
        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

    else:
        print('Invalid Dataset Input')

    # standardized the data
    sc_x = StandardScaler()
    sc_x.fit(X)
    X_std = sc_x.transform(X)

    sc_y = StandardScaler()
    sc_y.fit(y)
    y_std = sc_y.transform(y).flatten()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.3, random_state=1)

    # record initial time
    start = process_time()

    # regressor selection
    if args['regression_approach'] == "lr":
        # lr regressor
        regressor = mylr.lr.fit(X_train, y_train)


    elif args['regression_approach'] == "ransac":
        # ransac regressor
        regressor = myransac.ransac.fit(X_train, y_train)


    elif args['regression_approach'] == "ridge":
        # ridge regressor
        myridge.ridge.alpha = args['alpha1']
        myridge.ridge.solver = args['solver']
        regressor = myridge.ridge.fit(X_train, y_train)


    elif args['regression_approach'] == "lasso":
        # lasso regressor
        mylasso.lasso.alpha = args['alpha2']
        regressor = mylasso.lasso.fit(X_train, y_train)


    elif args['regression_approach'] == "rf":
        # rf regressor
        myrf.forest.n_estimators = args['n_estimators']
        myrf.forest.criterion = args['criterion']
        myrf.forest.n_jobs = args['n_jobs']
        regressor = myrf.forest.fit(X_train, y_train)

    elif args['regression_approach'] == "normal":
        # Normal Equation Solution regressor

        # use defined function to predict
        y_pred_train = mynormal.normal_equation(X_train, y_train)
        y_pred_test = mynormal.normal_equation(X_test, y_test)

        error_train = mean_squared_error(y_train, y_pred_train)
        error_test = mean_squared_error(y_train, y_pred_train)
        print('MSE: train: %.3f, test: %.3f' % (error_train, error_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        print('R^2: train: %.3f test: %.3f' % (r2_train, r2_train))
        sys.exit()


    elif args['regression_approach'] == "logistic":
        regressor = logistic_regression.logistic.fit(X_train, y_train)

    else:
        print('Invalid Regression Approach')

    # make predictions
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    # calculate the time elapse for each classifier
    elapse = process_time() - start

    # MSE calculation
    error_train = mean_squared_error(y_train, y_train_pred)
    error_test = mean_squared_error(y_test, y_test_pred)

    # r2 calculation
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print('\n#########  Result of {} approach ###############'.format(args['regression_approach']))
    print('--> Running time: ')
    print('The running time of {0} regressor is {1:.5f} s'.format(args['regression_approach'], elapse))
    print('--> MSE: ')
    print('[MSE] train: %.3f, test:%.3f' % (error_train, error_test))
    print('--> R^2: ')
    print('[R^2] train: %.3f, test:%.3f' % (r2_train, r2_test))
    print('####################  End ######################\n')





