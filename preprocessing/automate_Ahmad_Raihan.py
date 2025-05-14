import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocessing_pipeline(input=r'D:\6. Membangun Sistem Machine Learning\Eksperimen_SML_Ahmad-Raihan\dataset_raw\diamonds.csv', output=r'D:\6. Membangun Sistem Machine Learning\Eksperimen_SML_Ahmad-Raihan\preprocessing\diamond_preprocessing'):
  # Pastikan filenya sudah ada
  os.makedirs(output, exist_ok=True)
  
  # 1. Membaca file
  diamonds_df = pd.read_csv(input)

  # 2. Menghapus kolom yang tidak berguna
  diamonds_df_clean = diamonds_df.drop(columns=['Unnamed: 0', 'depth'])

  # 3. Menghapus nilai yang missing
  diamonds_df_clean = diamonds_df_clean.dropna()
  
  # 3. Menghapus data yang duplikat
  diamonds_df_clean = diamonds_df_clean.drop_duplicates()

  # 4. Menghapus outlier
  num_feature = diamonds_df_clean.select_dtypes(include='number')

  Q1 = num_feature.quantile(q=0.25)
  Q3 = num_feature.quantile(q=0.75)
  IQR = Q3-Q1
  BB = Q1 - (IQR * 1.5)
  BA = Q3 + (IQR * 1.5)

  mask_outlier = ((num_feature < BB) | (num_feature > BA)).any(axis=1)
  clean_index = num_feature.index[~mask_outlier]
  diamonds_df_clean = diamonds_df_clean.loc[clean_index]

  # 5. Menangani Data Skewed
  df_transformed = diamonds_df_clean.copy()
  cols_to_transform = ['carat', 'table', 'x', 'y', 'z', 'price']
  power_transformers = {}

  for col in cols_to_transform:
    transformer = PowerTransformer(standardize=True)
    y = np.asarray(df_transformed[col]).reshape(-1, 1)
    df_transformed[f'transform_{col}'] = transformer.fit_transform(y)
    df_transformed.drop(columns=[col], inplace=True)
    power_transformers[col] = transformer

  transform_diamonds_df_clean = df_transformed.copy()
  transform_diamonds_df_clean.head()

  # 6. Mengididentfikasi Fitur (X) dan Label (y)
  X = transform_diamonds_df_clean.drop(columns='transform_price')
  y = transform_diamonds_df_clean['transform_price']

  # 7. Membagi Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 8. Melakukan feature encoding
  encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
  X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[['cut', 'color', 'clarity']]), columns=encoder.get_feature_names_out(['cut', 'color', 'clarity']))
  X_test_encoded = pd.DataFrame(encoder.transform(X_test[['cut', 'color', 'clarity']]), columns=encoder.get_feature_names_out(['cut', 'color', 'clarity']))

  X_train_encoded = X_train_encoded.join(X_train.reset_index(drop=True))
  X_test_encoded = X_test_encoded.join(X_test.reset_index(drop=True))
  X_train_encoded.drop(columns=['cut', 'color', 'clarity'], inplace=True)
  X_test_encoded.drop(columns=['cut', 'color', 'clarity'], inplace=True)

  # 9. Simpan dataset hasil preprocessing
  X_train_encoded.to_csv(os.path.join(output, 'X_train.csv'), index=False)
  X_test_encoded.to_csv(os.path.join(output, 'X_test.csv'), index=False)
  y_train.to_csv(os.path.join(output, 'y_train.csv'), index=False)
  y_test.to_csv(os.path.join(output, 'y_test.csv'), index=False)

  # 10. Simpan proses preprocessing transformer dan encoder
  joblib.dump(power_transformers, os.path.join(output, 'power_transformers.joblib'))
  joblib.dump(encoder, os.path.join(output, 'onehot_encoder.joblib'))

  print(f'Proses preprorcessing telah selesai dan hasilnya disimpan di : {output}')

if __name__ == '__main__':
  preprocessing_pipeline()