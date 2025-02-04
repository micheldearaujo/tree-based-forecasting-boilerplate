import sys
sys.path.insert(0,'.')

data_config = {
  'paths': {
    'root_data_path': './data',
    'raw_data_path': 'data/raw',
    'processed_data_path': 'data/processed',
    'output_data_path': 'data/output',
    'models_path': './models'
  },

  'table_names': {
    'raw_table_name': 'raw_df.csv',
    'processed_table_name': 'featurized_df.csv',
    'output_table_name': 'forecast_output_df.csv',
    'model_performance_table_name': 'model_performance_daily.csv',
    'cross_validation_table_name': 'crossval_results.csv'
  }
}