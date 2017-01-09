# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

def import_data():
    # Read in csv file
    df = pd.read_csv('../Data/new_reggae_data.csv', header=0)
    return df

def preproc(df):
    # Preprocessing - remove all columns from data that we don't want
    # Aim is to only keep parameters from background fit - remove white noise
    drop_cols = ['KIC', 'anu', 'anu_err', 'DPi1', 'e_DPi1', 'q', 'Status', 'KALLINGER_EVSTATES', 'age', 'age_68L', 'age_68U', 'age_95L', 'age_95U', 'mass', 'mass_68L',
                 'mass_68U', 'mass_95L', 'mass_95U', 'logg', 'logg_68L', 'logg_68U', 'logg_95L', 'logg_95U', 'rad', 'rad_68L', 'rad_68U', 'rad_95L', 'rad_95U', 'logrho',
                 'logrho_68L', 'logrho_68U', 'logrho_95L', 'logrho_95U', 'mbol', 'mbol_68L', 'mbol_68U', 'mbol_95L', 'mbol_95U', 'Kepler', 'Kepler_68L',
                 'Kepler_68U', 'Kepler_95L', 'Kepler_95U', 'g', 'g_68L', 'g_68U', 'g_95L', 'g_95U', 'r', 'r_68L', 'r_68U', 'r_95L', 'r_95U',
                 'i', 'i_68L', 'i_68U', 'i_95L', 'i_95U', 'z', 'z_68L', 'z_68U', 'z_95L', 'z_95U', 'DDO51_finf', 'DDO51_finf_68L', 'DDO51_finf_68U',
                 'DDO51_finf_95L', 'DDO51_finf_95U', 'J', 'J_68L', 'J_68U', 'J_95L', 'J_95U', 'H', 'H_68L', 'H_68U', 'H_95L', 'H_95U', 'Ks', 'Ks_68L',
                 'Ks_68U', 'Ks_95L', 'Ks_95U', 'W1', 'W1_68L', 'W1_68U', 'W1_95L', 'W1_95U', 'W2', 'W2_68L', 'W2_68U', 'W2_95L', 'W2_95U', 'W3', 'W3_68L',
                 'W3_68U', 'W3_95L', 'W3_95U', 'W4', 'W4_68L', 'W4_68U', 'W4_95L', 'W4_95U', 'Av', 'Av_68L', 'Av_68U', 'Av_95L', 'Av_95U', 'mu0', 'mu0_68L',
                 'mu0_68U', 'mu0_95L', 'mu0_95U', 'dist', 'dist_68L', 'dist_68U', 'dist_95L', 'dist_95U', 'nfil', 'fils', 'Z', 'teff', 'feh', 'alpha/H',
                 'hsig1_err', 'b1_err','c1_err','hsig2_err', 'b2_err', 'c2_err', 'numax_err','denv_err','Henv_err', 'alpha_err','beta_err', 'c3_err', 'white', 'white_err',
                 'KEPLER_INT', '2MASS_ID', 'LOC_ID', 'RA', 'DEC', 'RA_PM', 'DEC_PM', 'UCAC_PM_RA', 'UCAC_PM_RA_ERR', 'UCAC_PM_DEC', 'UCAC_PM_DEC_ERR', 'G_MAG_KIC',
                 'R_MAG_KIC', 'I_MAG_KIC', 'Z_MAG_KIC', 'J_MAG_2M', 'J_MAG_ERR', 'H_MAG_2M', 'H_MAG_ERR', 'K_MAG_2M', 'K_MAG_ERR', 'U_SDSS', 'U_SDSS_err', 'G_SDSS',
                 'G_SDSS_err', 'R_SDSS', 'R_SDSS_err', 'I_SDSS', 'I_SDSS_err', 'Z_SDSS', 'Z_SDSS_err', 'G_SDSS_PIN', 'G_SDSS_PIN_ERR', 'R_SDSS_PIN', 'R_SDSS_PIN_ERR',
                 'I_SDSS_PIN', 'I_SDSS_PIN_ERR', 'Z_SDSS_PIN', 'Z_SDSS_PIN_ERR', 'WISE4_5', 'WISE4_5_ERR', 'KEP_MAG', 'KIC_TEFF', 'KIC_LOGG', 'KIC_FEH', 'KIC_EBMV',
                 'TEFF_SDSS', 'TEFF_SDSS_ERR', 'TEFF_IRFM', 'TEFF_IRFM_ERR', 'DR10_TEFF_FIT', 'DR10_TEFF_COR', 'DR10_TEFF_COR_ERR', 'DR10_TEFF_S2', 'DR10_LOGG_FIT',
                 'DR10_LOGG_COR', 'DR10_LOGG_COR_ERR', 'DR10_FE_H_FIT', 'DR10_FE_H_COR', 'DR10_FE_H_COR_ERR', 'CHAPLIN_NU_MAX', 'CHAPLIN_NU_MAX_ERR', 'CHAPLIN_DELTA_NU',
                 'CHAPLIN_DELTA_NU_ERR', 'CHAPLIN_LOGG', 'CHAPLIN_LOGG_PERR', 'CHAPLIN_LOGG_MERR', 'DR10_DELTA_NU', 'DR10_DELTA_NU_ERR', 'DR10_NU_MAX', 'DR10_NU_MAX_ERR',
                 'DR10_S1_MASS', 'DR10_S1_MASS_PERR', 'DR10_S1_MASS_MERR', 'DR10_S2_MASS', 'DR10_S2_MASS_PERR', 'DR10_S2_MASS_MERR', 'DR10_S1_RADIUS', 'DR10_S1_RADIUS_PERR',
                 'DR10_S1_RADIUS_MERR', 'DR10_S2_RADIUS', 'DR10_S2_RADIUS_PERR', 'DR10_S2_RADIUS_MERR', 'DR10_S1_LOGG', 'DR10_S1_LOGG_PERR', 'DR10_S1_LOGG_MERR', 'DR10_S2_LOGG',
                 'DR10_S2_LOGG_PERR', 'DR10_S2_LOGG_MERR', 'DR10_S1_DENSITY', 'DR10_S1_DENSITY_PERR', 'DR10_S1_DENSITY_MERR', 'DR10_S2_DENSITY', 'DR10_S2_DENSITY_PERR',
                 'DR10_S2_DENSITY_MERR', 'DR12_TEFF_FIT', 'DR12_TEFF_COR', 'DR12_TEFF_COR_ERR', 'DR12_LOGG_FIT', 'DR12_LOGG_COR', 'DR12_LOGG_COR_ERR', 'DR12_FE_H_ADOP_FIT',
                 'DR12_FE_H_ADOP_COR', 'DR12_FE_H_ADOP_COR_ERR', 'DR12_ALP_FE_ADOP_FIT', 'DR12_ALP_FE_ADOP_COR', 'DR12_ALP_FE_ADOP_COR_ERR', 'DR12_AL_H', 'DR12_AL_H_ERR',
                 'DR12_CA_H', 'DR12_CA_H_ERR', 'DR12_C_H', 'DR12_C_H_ERR', 'DR12_FE_H', 'DR12_FE_H_ERR', 'DR12_K_H', 'DR12_K_H_ERR', 'DR12_MG_H', 'DR12_MG_H_ERR', 'DR12_MN_H',
                 'DR12_MN_H_ERR', 'DR12_NA_H', 'DR12_NA_H_ERR', 'DR12_NI_H', 'DR12_NI_H_ERR', 'DR12_N_H', 'DR12_N_H_ERR', 'DR12_O_H', 'DR12_O_H_ERR', 'DR12_SI_H', 'DR12_SI_H_ERR',
                 'DR12_S_H', 'DR12_S_H_ERR', 'DR12_TI_H', 'DR12_TI_H_ERR', 'DR12_V_H', 'DR12_V_H_ERR', 'ASPCAP_CHI2', 'ASPCAP_SNR', 'VHELIO_AVG', 'VSCATTER', 'TEFF_FIT',
                 'TEFF_COR', 'TEFF_COR_ERR', 'LOGG_FIT', 'LOGG_COR', 'LOGG_COR_ERR', 'LGVMICRO', 'LGVSINI', 'FE_H_ADOP_COR', 'FE_H_ADOP_COR_ERR', 'ALP_FE_ADOP_COR',
                 'ALP_FE_ADOP_COR_ERR', 'AL_H', 'AL_H_ERR', 'CA_H', 'CA_H_ERR', 'CR_H', 'CR_H_ERR', 'C_H', 'C_H_ERR', 'FE_H', 'FE_H_ERR', 'K_H', 'K_H_ERR', 'MG_H',
                 'MG_H_ERR', 'MN_H', 'MN_H_ERR', 'NA_H', 'NA_H_ERR', 'NI_H', 'NI_H_ERR', 'N_H', 'N_H_ERR', 'O_H', 'O_H_ERR', 'SI_H', 'SI_H_ERR', 'S_H', 'S_H_ERR', 'TI_H',
                 'TI_H_ERR', 'V_H', 'V_H_ERR', 'VSINI_JT', 'VSINI_JT_ERR', 'TEFF_ROT_FIT', 'TEFF_ROT_FIT_ERR', 'FE_H_ADOP_ROT_FIT', 'FE_H_ADOP_ROT_FIT_ERR',
                 'ALPHA_FE_ADOP_ROT_FIT', 'ALPHA_FE_ADOP_ROT_FIT_ERR', 'NU_MAXDW', 'NU_MAXDW_STERR', 'NU_MAXDW_SYSERR', 'NU_MAXDW_TOTERR', 'DELTA_NUDW', 'DELTA_NUDW_STERR',
                 'DELTA_NUDW_SYSERR', 'DELTA_NUDW_TOTERR', 'NU_MAXRG', 'NU_MAXRG_ERR', 'NU_MAXRG_RANGE', 'LOGGRG', 'A2Z_NU_MAX', 'A2Z_NU_MAX_ERR', 'A2Z_DELTA_NU',
                 'A2Z_DELTA_NU_ERR', 'CAN_NU_MAX', 'CAN_NU_MAX_ERR', 'CAN_DELTA_NU', 'CAN_DELTA_NU_ERR', 'COR_NU_MAX', 'COR_NU_MAX_ERR', 'COR_DELTA_NU', 'COR_DELTA_NU_ERR',
                 'OCT_NU_MAX', 'OCT_NU_MAX_ERR', 'OCT_DELTA_NU', 'OCT_DELTA_NU_ERR', 'S1D_NU_MAX', 'S1D_NU_MAX_ERR', 'S1D_DELTA_NU', 'S1D_DELTA_NU_ERR', 'SYD_NU_MAX',
                 'SYD_NU_MAX_ERR', 'SYD_DELTA_NU', 'SYD_DELTA_NU_ERR', 'F8_LOGG', 'F8_LOGG_PERR', 'F8_LOGG_MERR', 'CHAPLIN', 'VANSADERS', 'ELSWORTH', 'STELLO_EVSTATES',
                 'MOSSER_EVSTATES', 'CONS_EVSTATES', 'EB_PER', 'N_KEP_QUART', 'CONSEC_3_QUART', 'OBS_QUART', 'TARGFLAGS', 'ASPCAPFLAGS', 'Kp']

    # Next set of columns to drop
    df = df.drop(drop_cols, axis=1)
    return df

def consensus_labelling(df):
    # Using KALLINGER_EVSTATES, CONS_EVSTATES, and evol_overall only use stars where
    # there is complete agreement between all methods
    df['KALLINGER_EVSTATES'][df['KALLINGER_EVSTATES'] == 3] = 0
    df['KALLINGER_EVSTATES'][df['KALLINGER_EVSTATES'] == -1] = -9999
    df = df[df['KALLINGER_EVSTATES'] != -9999]

    df['CONS_EVSTATES'][df['CONS_EVSTATES'] == 'RGB'] = 0
    df['CONS_EVSTATES'][df['CONS_EVSTATES'] == 'RGB/AGB'] = 0
    df['CONS_EVSTATES'][df['CONS_EVSTATES'] == 'RC'] = 1
    df['CONS_EVSTATES'][df['CONS_EVSTATES'] == 'RC/2CL'] = 2
    df['CONS_EVSTATES'][df['CONS_EVSTATES'] == '2CL'] = 2
    df = df[df['CONS_EVSTATES'] != '-9999']

    # Fetch indices where states are the same
    bigdiff = np.abs(np.diff(np.c_[df['evol_overall'],df['CONS_EVSTATES'], df['KALLINGER_EVSTATES']], axis=1))
    bigsum = np.sum(bigdiff, axis=1)
    df['bigsum'] = bigsum
    df = df[df['bigsum'] == 0]
    df.drop(['bigsum'], inplace=True, axis=1)

    return df

if __name__=="__main__":

    # See if argument given
    try:
        label_type = str(sys.argv[1])
    except:
        label_type = None

    # Import data
    df = import_data()

    # Choice as to whether you want to use consensus labels or those from Elsworth et al. (2017)
    if label_type == 'Cons':
        print("Using consensus labels")
        df = consensus_labelling(df)
    else:
        print("Choosing labels from Elsworth et al. (2017)")
    # Preprocessing
    df = preproc(df)
    # Create training set with known labels
    train = df[df['evol_overall'] != 3]
    # Create test set with unknown labels
    test = df[df['evol_overall'] == 3]

    test.to_csv('../Data/test_'+str(label_type)+'.csv', index=False)
    train.to_csv('../Data/train_'+str(label_type)+'.csv', index=False)
