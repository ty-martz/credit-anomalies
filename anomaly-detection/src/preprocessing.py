import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
import zipfile


def load_payments_from_zip(rel_path, defaul_csv='payments.csv'):
    with zipfile.ZipFile(rel_path, 'r') as zip_ref:
        # Loop through each file in the zip file
        print('available files:')
        for file_name in zip_ref.namelist():
            print(file_name)
        print('----')

        with zip_ref.open(defaul_csv) as file:
            df = pd.read_csv(file, parse_dates=['CHECK DATE'])
    
    return df


def get_more_info(df, cols=True):
    print(f'Dataset Size: {len(df)}')
    # duplicates
    num_duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {num_duplicates}")
    print('----')

    if cols:
        # column level info
        num_nulls = df.isnull().sum()
        pct_null = num_nulls / len(df) * 100
        unique_vals = df.nunique()
        print(r'Column Level Data:')
        print('----')
        for col, n, p, u in zip(df.columns, num_nulls, pct_null, unique_vals):
            print(f"{col}:")
            print(f"    Nulls = {n} ({round(p,2)}%)")
            print(f"    Unique = {u}")
    

def get_df_date_amt(df, show=True):
    ts = df.copy()[['CHECK DATE', 'AMOUNT']]
    ts.columns = ['date', 'amount']
    ts = ts.sort_values('date').reset_index(drop=True)
    if show:
        print(ts.head(3))
    return ts


def process_credit_data(df_og):
    df = df_og.copy()
    print('Removing dupes...')
    df = df.drop_duplicates()

    print('Breaking down dates...')
    df['month'] = df['CHECK DATE'].dt.month
    df['day'] = df['CHECK DATE'].dt.day
    df['year'] = df['CHECK DATE'].dt.year
    df['weekday'] = df['CHECK DATE'].dt.weekday

    # encode
    print('Encoding columns...')
    oe = OrdinalEncoder()
    df['voucher'] = oe.fit_transform(df[['VOUCHER NUMBER']])
    oe2 = OrdinalEncoder()
    df['contract'] = oe2.fit_transform(df[['CONTRACT NUMBER']])
    oe3 = OrdinalEncoder()
    df['vendid'] = oe3.fit_transform(df[['VENDOR NAME']])

    df['cashed_flag'] = df['CASHED'].map({True:1, False:0})

    df['DEPARTMENT NAME'] = df['DEPARTMENT NAME'].fillna('Unknown')
    dummies = pd.get_dummies(df['DEPARTMENT NAME'], prefix='dept')
    df = pd.concat([df, dummies], axis=1)

    print('Sorting transactions...')
    df.sort_values('CHECK DATE', inplace=True)

    print('Finalizing data...')
    df = df.drop(['VOUCHER NUMBER', 'CONTRACT NUMBER', 'VENDOR NAME', 'CHECK DATE', 'CASHED', 'DEPARTMENT NAME'], axis=1)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df = df.dropna().reset_index(drop=True)

    return df


def PCA_credit_data(n_comp, df, seed=31):

    pca = PCA(n_components=n_comp, random_state=seed)

    x_PCA = pca.fit_transform(df)
    x_PCA = pd.DataFrame(data=x_PCA, index=df.index)

    inv_PCA = pca.inverse_transform(x_PCA)
    inv_PCA = pd.DataFrame(data=inv_PCA, index=df.index)

    return x_PCA, inv_PCA
