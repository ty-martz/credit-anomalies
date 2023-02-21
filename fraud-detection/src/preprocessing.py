import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def split_data(df, target='Class', seed=31):
    x = df.copy().drop([target], axis=1)
    y = df[target].copy()

    xt, yt, xtest, ytest = train_test_split(x, y, test_size=0.2, random_state=seed)
    return (xt, yt, xtest, ytest)


def preprocess_fraud(df, pca_components=24, cols_to_scale=None, cols_to_keep=None):

    df_dict = {'clean': df[df['Amount'] > 0].copy().reset_index(drop=True)} # to hold different cleaned dfs

    # standard scaling
    cleaner = df[df['Amount'] > 0].copy() # remove worthless transactions

    if cols_to_scale is None:
        cols_to_scale = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    if cols_to_keep is None:
        cols_to_keep = ['Time', 'Amount', 'Class']

    scaler = StandardScaler()
    cleaner[cols_to_scale] = scaler.fit_transform(cleaner[cols_to_scale])
    df_dict['scaled'] = cleaner.reset_index(drop=True)

    # pca
    pca_prep = df[df['Amount'] > 0].copy()
    pca_mod = PCA(pca_components)
    upd = pca_mod.fit_transform(pca_prep[cols_to_scale])
    new_cols = pd.DataFrame(upd)
    new_cols.columns = [f"V{i}" for i in new_cols.columns]

    pca_df = pd.concat([pca_prep[cols_to_keep].copy(), new_cols], axis=1)
    scaler2 = StandardScaler()
    pca_df[new_cols.columns] = scaler2.fit_transform(pca_df[new_cols.columns])
    
    df_dict['pca'] = pca_df.dropna().reset_index(drop=True)

    return df_dict