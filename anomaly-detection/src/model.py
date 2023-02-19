import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from preprocessing import PCA_credit_data


def score_anomaly(df, pca_df):
    loss = np.sum((np.array(df) - np.array(pca_df))**2, axis=1) # sum squared differences
    loss = pd.Series(data=loss, index=df.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss)) # scaling to be between 0 and 1

    return loss


def show_reconstruction(loss, thresh=0.5):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(loss.index, loss)
    ax.axhline(y=thresh, color='b')
    ax.set_xlabel('Index')
    ax.set_ylabel('Score')
    ax.set_title(f'PCA Reconstruction ({thresh} Threshold)')
    plt.show()


def load_clean_data(path='../data/clean-credit.pkl'):
    return pd.read_pickle(path)


def main(n_PCA_components=32, thresh=0.5):
    clean = load_clean_data()
    pca1, pca_inv = PCA_credit_data(n_PCA_components, clean)
    loss = score_anomaly(clean, pca_inv)
    anoms = len(np.where(loss > thresh)[0])

    # show results:
    print('########')
    print('PCA Results:')
    print('----')
    print(f'Input Components = {n_PCA_components}')
    print(f'Reconstruction Threshold = {thresh}')
    print('')
    print(f'{anoms} Anomalies found')
    print(f'{round((anoms / len(clean) * 100), 2)} % anomaly rate across all transactions')
    show_reconstruction(loss, thresh)
    

if __name__ == '__main__':
    main()
