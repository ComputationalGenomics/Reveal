__author__ = "Filippo Utro"
__copyright__ = "Copyright 2019, IBM Research"

__version__ = "0.0.1"
__maintainer__ = "Filippo Utro"
__email__ = "futro@us.ibm.com"
__status__ = "Development"

import argparse
import logging
import math
import os
from random import shuffle
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def build_parser():
    parser = argparse.ArgumentParser(description="ReVeaL: Rare Variant Learning")
    parser.add_argument('--sample_info', '-s', type=str, required=True,
                        help="File containing the samples alteration")
    parser.add_argument('--label_info', '-l', type=str, required=True,
                        help="File containing the label for each samples")
    parser.add_argument('--num_fold', '-nf', type=int, required=False, default=10,
                        help="Number of fold to be generated")
    parser.add_argument('--sample_size', '-ns', type=int, required=False, default=35,
                        help="Number of samples to be selected for the shingle construction")
    parser.add_argument('--test_train', '-tt', type=str, required=True,
                        help="File containing the train/test number ratio")
    parser.add_argument('--pre_computed', '-pre', default=None, required=False,
                        help="Indicated if the ")
    parser.add_argument('--window_size', '-w', type=int, required=False, default=-1,
                        help="Size of the window for the mutation load computation, if not provided the full "
                                    "length of each region (see regions_size option) is used ")
    parser.add_argument('--permuted', '-p', default=False, action="store_true", required=False)
    parser.add_argument('--moment', '-m', type=int, required=False, default=1, choices=range(1, 4),
                        help="Moment to compute for the shingle construction (default value 1 (mean))")
    parser.add_argument('--regions_size', '-r', type=str, required=True,
                        help="File containing the region's size per row")
    parser.add_argument('--store_out_folder', '-o', type=str, required=True, help='Name of the output folder.')

    if len(sys.argv) < 7:
        parser.print_help(sys.stderr)
    return parser.parse_args()


def _compute_count_in_region(sample_data, start, stop):
    a = sample_data[(sample_data['start'] >= start) & (sample_data['stop'] <= stop)]  # fully contained
    b = sample_data[(sample_data['stop'] > start) & (sample_data['stop'] <= stop)]  # partially contained
    c = sample_data[(sample_data['start'] >= start) & (sample_data['start'] <= stop)]  # partially contained
    #d = sample_data[(sample_data['start'] >= start) & (sample_data['stop'] >= stop)]  # partially contained
    a = a.append(b).append(c).drop_duplicates() #.append(d)
    return a.groupby('samples').count().reset_index()


def create_prep_files(sample_data, samples, regions_size, out_folder, chromosome, window_size=-1):
    """
        Function to create the mutational load files
        :parameter sample_data: panda data frame containing the sample with relative start and stop alteration
        :parameter samples: list of samples
        :parameter regions_size: panda data frame containing the start and stop of the region of interest
        :parameter out_folder: folder where to store the output files
        :parameter chromosome: integer relative to the chromosome of interest
        :parameter window_size: integer relative the the window size
    """
    mutation_load_data = pd.DataFrame()
    for index_region, row_region in regions_size.iterrows():
        start = row_region['start']
        stop = row_region['stop']

        if window_size > 0:
            for bin_window in range(0, math.ceil((stop-start)/window_size)):
                end = end + (bin_window * window_size)
                count_in_window = _compute_count_in_region(sample_data, start, end)
                mutation_load_data_bin = pd.DataFrame()
                mutation_load_data_bin['samples'] = count_in_window['samples']
                mutation_load_data_bin['count'] = count_in_window['start']
                missing_sample_in_window = set(samples).difference((set(mutation_load_data_bin['samples'])))
                tmp_data = pd.DataFrame({'sample': list(missing_sample_in_window),
                                         'count': [0]*len(missing_sample_in_window)})
                
                mutation_load_data_bin = pd.concat([mutation_load_data_bin, tmp_data])
                mutation_load_data_bin['window'] = [(chromosome, start, stop, bin_window)] * len(mutation_load_data_bin)
                mutation_load_data = pd.concat([mutation_load_data, mutation_load_data_bin])
        else:
            count_in_window = _compute_count_in_region(sample_data, start, stop)
            mutation_load_data['samples'] = count_in_window['samples']
            mutation_load_data['count'] = count_in_window['start']
            missing_sample_in_window = set(samples).difference((set(mutation_load_data['samples'])))
            tmp_data = pd.DataFrame({'sample': list(missing_sample_in_window),
                                     'count': [0] * len(missing_sample_in_window)})
            mutation_load_data = pd.concat([mutation_load_data, tmp_data])
            mutation_load_data['window'] = [(chromosome, start, stop, -1)] * len(mutation_load_data)

    mutation_load_pivot = mutation_load_data.sort_values(['samples', 'window']).pivot_table('samples', 'window',
                                                                                            'count').fillna(0)
    mutation_load_pivot.to_csv(os.path.join(out_folder, 'mutation_load'+str(chromosome)+'.csv'))
    return mutation_load_pivot


def _compute_shingle(mutation_load, list_samples, id_sample, moment_type=1):
    unique, counts = np.unique(list_samples, return_counts=True)
    tmp = mutation_load.loc[unique, :].copy()
    for i in range(0, len(counts)):
        if counts[i] > 1:
            tmp = tmp.append(mutation_load.loc[unique[i], :], sort=False)
    
    if (moment_type == 1):
        moment1 = pd.DataFrame(tmp.mean())
    elif (moment_type == 2):
        moment1 = pd.DataFrame(tmp.var())
    elif (moment_type == 3):
        moment1 = pd.DataFrame(tmp.skew())
    else:
        moment1 = pd.DataFrame(tmp.kurt())
    moment1['samples'] = id_sample

    return moment1


def generate_train_test(train_samples, test_samples, store_out_folder, pre_computed, chromosome, moment_type):
    """
    This function compute the shingles, then generating the train and test sample
    :param train_samples:
    :param test_samples:
    :param store_out_folder:
    :param pre_computed:
    :param chromosome:
    :return:
    """
    if pre_computed is not None:
        path_pre = pre_computed
    else:
        path_pre = store_out_folder

    mutation_load = pd.read_csv(os.path.join(path_pre,  'mutation_load' + str(chromosome) + '.csv'))
    shingles = pd.DataFrame()

    for key, samples_in_train in train_samples.Keys():
        id_sample = 'Train_'+str(key[1])+'_fold_'+str(key[0])+'_'+key[2]
        moment1 = _compute_shingle(mutation_load, samples_in_train, id_sample, moment_type)
        shingles = shingles.append(moment1.T)

    for key, samples_in_test in test_samples.Keys():
        id_sample = 'Test_'+str(key[1])+'_fold_'+str(key[0])+'_'+key[2]
        moment1 = _compute_shingle(mutation_load, samples_in_test, id_sample, moment_type)
        shingles = shingles.append(moment1.T)

    shingles.to_csv(os.path.join(store_out_folder, 'shingle_'+str(chromosome)+'.csv'))

    return shingles


if __name__ == "__main__":
    args = build_parser()

    folder = os.path.dirname(os.path.abspath(__file__))
    store_out_folder = os.path.join(folder, args.store_out_folder)

    if not os.path.exists(store_out_folder):
        os.makedirs(store_out_folder, exist_ok=True)

    logging.basicConfig(filename=os.path.join(store_out_folder, 'ReVeaL.log'), level=logging.INFO)

    if args.pre_computed is not None:
        logging.info('Analyzing sample_file: %s', args.sample_info, )

        sample_info = pd.read_csv(args.sample_info, sep='\t')
        chromosomes = sample_info['chr'].unique()
        regions = pd.read_csv(args.regions_size, sep='\t')

        Parallel(n_jobs=22, verbose=5)(delayed(create_prep_files)(sample_info[sample_info['chr'] == chr_interest], sample_info['samples'].unique(),
                                                                  regions[regions['chr'] == chr_interest],
                                                                  store_out_folder, chr_interest, args.window_size
                                                                  ) for chr_interest in chromosomes)

    chrs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22']

    label_info = pd.read_csv(args.label_info)
    test_train_sizes = pd.read_csv(args.test_train)

    train_samples = {}
    test_samples = {}
    for fold in range(0, args.num_fold):
        tmp_phenotype = label_info['phenotype'].tolist()
        if args.permuted:
            logging.info('Permuting labels')
            shuffle(tmp_phenotype)
            label_info['phenotype'] = tmp_phenotype
        for index_tr, row_tr in test_train_sizes.iterrows():
            size_train = row_tr['Train']
            size_test = row_tr['Test']
            samples = label_info[label_info['phenotype'] == row_tr['phenotype']]['samples'].tolist()
            for i in range(0, size_train):
                selected_samples = np.random.choice(samples, args.sample_size, replace=True)
                train_samples[(fold, i, row_tr['phenotype'])] = selected_samples
            for i in range(0, size_test):
                selected_samples = np.random.choice(samples, args.sample_size, replace=True)
                test_samples[(fold, i, row_tr['phenotype'])] = selected_samples

    r = Parallel(n_jobs=22, verbose=5)(delayed(generate_train_test)(train_samples, test_samples,
                                                                    store_out_folder, args.pre_computed, 
                                                                    chr_interest, args.moment)
                                       for chr_interest in chrs)
