import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self.load_file()

    def load_file(self) -> pd.DataFrame:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(self.file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(self.file_path)
        elif ext == '.parquet':
            return pd.read_parquet(self.file_path)
        elif ext == '.feather':
            return pd.read_feather(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def basic_info(self):
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())
        print("\nNull values:")
        print(self.df.isnull().sum())

    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        return self.df

    def impute_missing(self, method: str = 'mean'):
        for col in self.df.select_dtypes(include='number'):
            if self.df[col].isnull().any():
                if method == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif method == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
        return self.df

    def binarize_column(self, column: str, threshold: Union[int, float]):
        self.df[column + '_binary'] = (self.df[column] > threshold).astype(int)
        return self.df

    def basic_eda(self):
        print("\nDescriptive Statistics:")
        print(self.df.describe(include='all'))

        print("\nCorrelation Matrix:")
        corr = self.df.select_dtypes(include='number').corr()
        print(corr)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

        return corr

    def univariate_analysis(self):
        for col in self.df.select_dtypes(include=np.number).columns:
            plt.figure()
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Histogram for {col}")
            plt.show()

    def bivariate_analysis(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                plt.figure()
                sns.scatterplot(x=self.df[col1], y=self.df[col2])
                plt.title(f"Scatter Plot: {col1} vs {col2}")
                plt.show()

    def detect_outliers(self):
        outliers = {}
        for col in self.df.select_dtypes(include=np.number):
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers[col] = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        print("\nOutlier Summary:")
        print(outliers)
        return outliers

    def get_llm_summary(self):
        summary = {}

        def stringify_dict_keys(d):
            return {str(k): v for k, v in d.items()}

        summary['columns'] = [{'name': col, 'dtype': str(dtype)} for col, dtype in self.df.dtypes.items()]
        summary['missing_values'] = self.df.isnull().sum()[lambda x: x > 0].sort_values(ascending=False).to_dict()

        try:
            describe = self.df.describe(include='all', datetime_is_numeric=True).transpose().fillna("").to_dict()
        except TypeError:
            describe = self.df.describe(include='all').transpose().fillna("").to_dict()
        summary['describe'] = describe

        corr = self.df.select_dtypes(include='number').corr()
        top_corrs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)) \
                        .stack().sort_values(ascending=False).head(5).to_dict()
        summary['top_correlations'] = stringify_dict_keys(top_corrs)

        summary['top_categories'] = {
            col: self.df[col].value_counts().head(5).to_dict()
            for col in self.df.select_dtypes(include='object').columns
        }

        summary['unique_counts'] = stringify_dict_keys({col: self.df[col].nunique() for col in self.df.columns})

        return summary
