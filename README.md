# Payments Fraud and Anomaly Detection
----
Using unsupervised and supervised learning methods to detect fraud and anomalies in credit card data. When all you know are simple transactiondetails of a purchase

----
### Datasets:
1. [Credit Card Fraud (labeled)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    - This dataset contains credit card transactions made by European cardholders over a period of two days in September 2013. It has a total of 284,807 transactions, out of which 492 are fraud.
2. [Chicago Vendor Payments](https://www.kaggle.com/datasets/chicago/chicago-payments?resource=download)
    - Public data showing all vendor payments made by the City of Chicago from 1996 to present

----
### Project Structure

```bash
├── data
│   ├── holds raw and cleaned data
├── notebooks
│   ├── EDA and preprocessing
│   ├── Model building and training
├── src
│   ├── preprocessing
│   ├── modeling
```

- The top folders hold a structure similar to above, but each with a focus
    - `fraud-detection` focuses on the labeled data and uses machine learning models to determine if a purchase is fraudulent
    - `anomaly-detecton` uses unsupervised learning on the Chicago dataset to find anomalies in the vendor payments
- The `data` directory contains the raw and cleaned datasets
- The `notebooks` directory contains Jupyter notebooks for performing exploratory data analysis (EDA), preprocessing, and modeling.
- The `src` directory contains Python modules for preprocessing and modeling.

