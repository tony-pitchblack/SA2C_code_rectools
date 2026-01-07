## Copy YooChoose/Retailrocket datasets from unzipped `SA2C_code.zip`
SA2C_code_unzip_dir=/raid/data_share/antonchernov/transformer_benchmark_rl/data/raw/SA2C_code
rsync -a "${SA2C_code_unzip_dir}/Kaggle/data/" "Kaggle/data/"
rsync -a "${SA2C_code_unzip_dir}/RC15/data/" "RC15/data/"