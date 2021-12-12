"""
Preprocessing functions
@Author: WangZihan05
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


NUM_BUCKET = 10
PARSE_DIR = "/data1/jiayu_xiao/project/wzh/Multitask/data/"

select_columns = ['clicked','followed','fan_count','upload_photo_cnt_7d', 'upload_photo_cnt_30d',
       'accu_public_visible_photo_cnt', 'combo_user_uv_1d',
       'combo_user_uv_7d', 'combo_user_click_1d', 'combo_user_click_7d',
       'combo_user_follow_1d', 'combo_user_follow_7d', 'usertab_user_uv_1d',
       'usertab_user_uv_7d', 'usertab_user_click_1d', 'usertab_user_click_7d',
       'usertab_user_follow_1d', 'usertab_user_follow_7d',
       'combo_keyword_uv_1d', 'combo_keyword_uv_7d', 'user_keyword_uv_1d',
       'user_keyword_uv_7d', 'combo_keyword_user_uv_1d',
       'combo_keyword_user_uv_7d', 'combo_keyword_user_click_1d',
       'combo_keyword_user_click_7d', 'combo_keyword_user_follow_1d',
       'combo_keyword_user_follow_7d', 'user_keyword_user_uv_1d',
       'user_keyword_user_uv_7d', 'user_keyword_user_click_1d',
       'user_keyword_user_click_7d', 'user_keyword_user_follow_1d',
       'user_keyword_user_follow_7d', 'relevance_score',
       'combo_keyword_user_ctr_1d', 'combo_keyword_user_ctr_7d',
       'combo_keyword_user_ftr_1d', 'combo_keyword_user_ftr_7d',
       'user_keyword_user_ctr_1d', 'user_keyword_user_ctr_7d',
       'user_keyword_user_ftr_1d', 'user_keyword_user_ftr_7d']

IGN_COL = ['clicked','followed']

# Feature Dictionary of click through
class FeatureDictionary(object):
    def __init__(self,
                 df_train,
                 df_test,
                 df_val):

        self.dfTrain = df_train
        self.dfTest = df_test
        self.dfVal = df_val

        self.feat_dim = 0
        self.feat_dict = {}

        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.dfTrain, self.dfTest, self.dfVal], sort=False)
        tc = 0
        for col in df.columns:
            if col in IGN_COL:
                continue
            else:
                us = df[col].unique()  # unique sample
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc

    def parse(self, df=None):

        if not self.feat_dict:
            raise ValueError("feat_dict is empty!!")

        dfi = df.copy()

        y = dfi[IGN_COL]
        dfi.drop(IGN_COL, axis=1, inplace=True)

        for col in dfi.columns:
            if col in IGN_COL:
                dfi.drop(col, axis=1, inplace=True)
            else:
                out = open("feat_dict","w")
                out.write(str(self.feat_dict))
                dfi[col] = dfi[col].map(self.feat_dict[col])
        return dfi, y


def parse_kwai():

    input_train_dir = "data/kwai/train_select.csv"

    print("\tLoading dataset ...")
    df_select = pd.read_csv(input_train_dir)[select_columns]
    df_select = df_select[(df_select["clicked"]==0)|(df_select["clicked"]==1)]
    df_select = df_select[(df_select["followed"]==0)|(df_select["followed"]==1)]
    df_train_select = df_select[:int(0.8*len(df_select))]
    df_test_select = df_select[int(0.8*len(df_select)):]

    num_col = select_columns[2:]
    ################ handle Missing values ##################

    print("\tFixing missing values ...")
    df_train = _fix_missing_values(df_train_select)
    train_len = len(df_train)
    df_test = _fix_missing_values(df_test_select)

    print("\tNormalizing numerical features ...")
    df_all = df_train.append(df_test)
    df_all = _norm_bucket_numerical(df_all, num_col)


    # split train, valid, and test
    print("\tSplitting Train, Valid, and Test Dataset ...")
    df_train = df_all[:train_len]
    df_test = df_all[train_len:train_len+6000]
    df_val = df_all[train_len+6000:]

    # split ind, val, and test
    print("\tSplitting Index, Value, and Labels ...")
    full_splits = _split_ind_val_label(dataset="kwai",
                                       df_train=df_train,
                                       df_test=df_test,
                                       df_val=df_val)

    # save 3X3 dataframes to `parsed` folder
    print("\tSaving all splited matrices ...")
    _save_splits(full_splits, dataset="kwai")



def _fix_missing_values(df):
    nan_convert_map = {
        'int64': 0,
        'float64': 0.0,
        'O': "00000000",
        'object': "00000000"
    }
    for col in df.columns:
        patch = nan_convert_map[str(df[col].dtype)]
        df[col] = df[col].fillna(patch)

    return df



def _split_ind_val_label(dataset, df_train, df_test, df_val):
    feat_dict = FeatureDictionary(df_train=df_train,
                                  df_test=df_test,
                                  df_val=df_val)

    # parse datasets

    df_train_split = feat_dict.parse(df=df_train)
    df_test_split = feat_dict.parse(df=df_test)
    df_val_split = feat_dict.parse(df=df_val)

    return df_train_split, df_val_split, df_test_split, feat_dict


def _save_splits(splits, dataset):

    usage = ["train", "val", "test"]
    term = ["ind", "label"]

    if not os.path.isdir(PARSE_DIR + dataset):
        os.mkdir(PARSE_DIR + dataset)

    for i, u in enumerate(usage):
        for j, t in enumerate(term):
            splits[i][j].to_csv(
                PARSE_DIR + "{}/{}_{}.csv".format(dataset, u, t),
                index=False,
                header=False
            )

    with open(PARSE_DIR + "{}/feat_dict".format(dataset), "w") as fout:
        # feature size field size
        fout.write("{} {}".format(splits[3].feat_dim, splits[0][0].shape[1]))


def _norm_bucket_numerical(df, num_col):
    """bucketing numerical features
    """
    for column in num_col:
        print(column)
        df[column] = pd.qcut(df[column],NUM_BUCKET,duplicates="drop")
    return df


if __name__ == "__main__":
    parse_kwai()
    print("finish!")