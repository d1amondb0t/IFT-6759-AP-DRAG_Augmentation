

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

cols = ['id-01', 'id-02', 'id-03', 'id-04', 'id-05', 'id-06', 'id-07', 'id-08', 'id-09',
        'id-10', 'id-11', 'id-12', 'id-13', 'id-14', 'id-15', 'id-16', 'id-17', 'id-18',
        'id-19', 'id-20', 'id-21', 'id-22', 'id-23', 'id-24', 'id-25', 'id-26', 'id-27',
        'id-28', 'id-29', 'id-30', 'id-31', 'id-32', 'id-33', 'id-34', 'id-35', 'id-36',
        'id-37', 'id-38']

cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 
            'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

def _load(path:str) -> pd.DataFrame: 
  return pd.read_csv(path)

def _merge(transaction: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
  return transaction.merge(identity, on="TransactionID", how="left")

def _rename_id_cols(test: pd.DataFrame) -> pd.DataFrame:
  for col in cols:
    test.rename(columns={col: col.replace('-','_')}, inplace=True)
  return test

def _create_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
  train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
  train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
  train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

  test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
  test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
  test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
  test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

  train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
  train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
  train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
  train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

  test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
  test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
  test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
  test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

  train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
  train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
  train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
  train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

  test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
  test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
  test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
  test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

  train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
  train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
  train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
  train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

  test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
  test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')
  test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
  test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')
  
  train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
  train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
  test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
  test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)
  
  return train, test

def _drop_low_info_cols(train: pd.DataFrame, test:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
  one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

  many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
  many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

  big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
  big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

  cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))
  ##cols_to_drop.remove('isFraud')
  cols_to_drop = [c for c in cols_to_drop if c != 'isFraud']

  train = train.drop(cols_to_drop, axis=1)
  test = test.drop(cols_to_drop, axis=1)
  
  return train, test

def _label_encode(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  for col in cat_cols:
    if col in train.columns:
      le = LabelEncoder()
      le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
      train[col] = le.transform(list(train[col].astype(str).values))
      test[col] = le.transform(list(test[col].astype(str).values))
  return train, test
      
def _sort_by_time(train: pd.DataFrame) -> pd.DataFrame:
    return train.sort_values('TransactionDT').reset_index(drop=True)

def _clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
  return df.replace([np.inf, -np.inf], np.nan)

def run_pipeline(
  transaction_path: str,
  identity_path: str,
  test_transaction_path: str,
  test_identity_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  
  train_tr = _load(transaction_path)
  train_id = _load(identity_path)
  test_tr  = _load(test_transaction_path)
  test_id  = _load(test_identity_path)
  
  train = _merge(train_tr, train_id)
  test  = _merge(test_tr,  test_id)

  test  = _rename_id_cols(test)

  train, test = _create_features(train, test)
  train, test = _drop_low_info_cols(train, test)
  train, test = _label_encode(train, test)

  train = _sort_by_time(train)

  train = _clean_inf_nan(train)
  test  = _clean_inf_nan(test)

  return train, test