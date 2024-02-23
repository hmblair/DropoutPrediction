# empirical_bayes.py

import numpy as np
import pandas as pd
import xarray as xr
from lib.models.baselines import BayesEstimator
from .utils import longest_run, longest_hairpin

# this function will be used to compute the features used to fit the empirical
# bayes estimator. modify it as needed, such as to use the longest_run function
# instead of longest_hairpin.
def compute_features(
        sequences : np.ndarray, 
        structures : np.ndarray, 
        ) -> np.ndarray:
    return np.array(
        [longest_hairpin(x) for x in structures]
        )

def load_reads_compute_features(
        file : str, 
        normalise : bool = True,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loda the 2A3 and DMS reads, sequences, and structures, from a netcdf file.
    """
    arr = xr.load_dataset(file)
    reads_2A3 = arr['reads_2A3'].values
    reads_DMS = arr['reads_DMS'].values
    if normalise:
        reads_2A3 = reads_2A3 / np.mean(reads_2A3) * 1000
        reads_DMS = reads_DMS / np.mean(reads_DMS) * 1000
    sequences = arr['sequence'].values
    structures = arr['structure'].values
    features = compute_features(sequences, structures)
    return reads_2A3, reads_DMS, features


def predict(
        reads : np.ndarray, 
        features : np.ndarray,
        estimator : BayesEstimator,
        normalise : bool = True,
        ) -> tuple[float, float, float]:
    """
    Predict the reads using the empirical bayes estimator, and compute the loss,
    correlation, and absolute error.

    Parameters
    ----------
    reads : np.ndarray
        The ground truth number of reads.
    features : np.ndarray
        The features to use for prediction.
    estimator : BayesEstimator
        The empirical bayes estimator to use for prediction.
    normalise : bool
        Whether to normalise the predictions and reads to have the same mean 
        before computing the loss, correlation, and absolute error.
    """
    # predict the log-reads
    preds, _ = estimator.predict(features)
    # exponentiate the predictions
    preds = np.exp(preds)

    # normalise, if desired
    if normalise:
        preds = preds / np.mean(preds) * 1000
        reads = reads / np.mean(reads) * 1000

    # compute the loss, correlation, and absolute error
    loss = np.sqrt(np.mean((np.log10(preds) - np.log10(reads))**2))
    correlation = np.corrcoef(np.log10(preds), np.log10(reads))[0, 1]
    absolute_error = np.mean(np.abs(preds - reads_2A3))

    return loss, correlation, absolute_error


# Fit the empirical bayes models
def fit(
        features : np.ndarray,
        reads_2A3 : np.ndarray,
        reads_DMS : np.ndarray,
        ) -> tuple[BayesEstimator, BayesEstimator]:
    """
    Fit the empirical bayes models for 2A3 and DMS reads.

    Parameters
    ----------
    features : np.ndarray
        The features to use for prediction.
    reads_2A3 : np.ndarray
        The ground truth number of 2A3 reads.
    reads_DMS : np.ndarray
        The ground truth number of DMS reads.
    """
    # fit the empirical bayes model for log 2A3 reads
    eb_2A3 = BayesEstimator()
    eb_2A3.fit(
        features,
        np.log(reads_2A3), 
        )
    # fit the empirical bayes model for log DMS reads
    eb_DMS = BayesEstimator()
    eb_DMS.fit(
        features,
        np.log(reads_DMS), 
        )
    return eb_2A3, eb_DMS


if __name__ == '__main__':  
    # Load the train data and compute the features
    reads_2A3, reads_DMS, features = load_reads_compute_features(
        'data/onemil1_1.nc'
        )

    # Fit the empirical bayes models
    eb_2A3, eb_DMS = fit(features, reads_2A3, reads_DMS)

    # compute predictions on the test datasets
    for dataset in ['onemil2', 'p240', 'p390']:
        # load the data
        reads_2A3, reads_DMS, features = load_reads_compute_features(
            f'data/{dataset}.nc'
            )

        # compute the loss for 2A3
        loss_2A3, correlation_2A3, absolute_error_2A3 = predict(
            reads_2A3, features, eb_2A3
            )

        # compute the loss for DMS
        loss_DMS, correlation_DMS, absolute_error_DMS = predict(
            reads_DMS, features, eb_DMS
            )

        # store the results in a dataframe
        df = pd.DataFrame({
            '2A3 Loss': [loss_2A3],
            '2A3 Correlation': [correlation_2A3],
            '2A3 Absolute error': [absolute_error_2A3],
            'DMS Loss': [loss_DMS],
            'DMS Correlation': [correlation_DMS],
            'DMS Absolute error': [absolute_error_DMS],
            })
        
        # round the results
        df = df.round(
            {
                '2A3 Loss': 4,
                '2A3 Correlation': 4,
                '2A3 Absolute error': 0,
                'DMS Loss': 4,
                'DMS Correlation': 4,
                'DMS Absolute error': 0,
            }
        )
        
        # print the results
        print(f'Results for {dataset}:')
        print(df)
        print()