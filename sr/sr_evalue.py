import argparse
import glob
import logging
import multiprocessing as mp
import os
import pickle
import time



import numpy as np
import pandas as pd

from sr.spectral_residual import SpectralResidual
#from msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode


# logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def run_spectral_residual(data):
    ts_data, params = data
    detector = SpectralResidual(series=ts_data,
                                sensitivity=99,  # only applies to margin
                                # threshold=0.1,
                                # mag_window=MAG_WINDOW,
                                # score_window=SCORE_WINDOW,
                                # batch_size=-1
                                **params)
    ts_result = detector.detect()
    return ts_result


def sr(dataset):
    parser = argparse.ArgumentParser(description='SR Evaluate')
    parser.add_argument('--csv-input-dir', type=str, default='..\\kpi\\kpi_test_ts_data', help='Dataset CSV input dir')
    parser.add_argument('--parallel', action='store_true', help='Run model in parallel')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.csv_input_dir)

    #dataset = np.load(Pathdataset)


    def objective(hyperopt_params):
        run_parallel = hyperopt_params.pop("run_parallel")
        data_with_params = [(dataset[:,i], hyperopt_params) for i in range(dataset.shape[1])]
        if run_parallel:
            results = pool.map(run_spectral_residual, data_with_params)

        else:
            results = [run_spectral_residual(ts_data_params) for ts_data_params in data_with_params]
            t = np.stack(results).T
        return t


    if args.parallel:
        pool = mp.Pool(processes=4)

    sets = objective({'batch_size': -1, 'mag_window': 3, 'score_window': 10000, 'threshold': 0.375, 'run_parallel': args.parallel})
    #np.save('..\\data\\SMD\\SMD_testsr.npy',sets)
    return sets
