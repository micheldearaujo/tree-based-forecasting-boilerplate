# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.utils.dates import days_ago
# from datetime import timedelta
# import subprocess 

# # Define default arguments for the DAG
# default_args = {
#     'owner': 'Michel',
#     'depends_on_past': False,  # Tasks don't depend on previous runs
#     'email': ['michelarrudala@gmail.com.com'],
#     'email_on_failure': True,
#     'email_on_retry': True,
#     'retries': 2,               # Number of retries if a task fails
#     'retry_delay': timedelta(minutes=10)  # Delay between retries
# }

# # Define the DAG 
# dag = DAG(
#     'daily_forecasting_pipeline',
#     default_args=default_args,
#     description='Daily pipeline for data processing, feature engineering, model training and prediction',
#     schedule_interval='0 11 * * 1-5',  # Run on work-days at 11am
#     start_date=days_ago(1),  # Start date (yesterday)
#     catchup=False  # Don't run for missed schedules
# )

# # Define task functions (use subprocess to call your scripts)
# def run_make_dataset():
#     subprocess.run(["python", "scripts/run_data_aquisition.py"])

# def run_feat_eng():
#     subprocess.run(["python", "scripts/run_feature_engineering.py"])

# def run_daily_evaluation():
#     subprocess.run(["python", "scripts/run_evaluation.py"])

# def run_train_model():
#     subprocess.run(["python", "scripts/run_model_training.py"])

# def run_predict_model():
#     subprocess.run(["python", "scripts/run_inference.py"])

# # Create tasks using PythonOperator
# make_dataset_task = PythonOperator(
#     task_id='make_dataset',
#     python_callable=run_make_dataset,
#     dag=dag
# )

# feat_eng_task = PythonOperator(
#     task_id='feature_engineering',
#     python_callable=run_feat_eng,
#     dag=dag
# )

# evaluation_task = PythonOperator(
#     task_id='daily_evaluation',
#     python_callable=run_daily_evaluation,
#     dag=dag
# )

# train_model_task = PythonOperator(
#     task_id='train_model',
#     python_callable=run_train_model,
#     dag=dag
# )

# predict_model_task = PythonOperator(
#     task_id='predict_model',
#     python_callable=run_predict_model,
#     dag=dag
# )

# # Define task dependencies
# make_dataset_task >> feat_eng_task >> evaluation_task >> train_model_task >> predict_model_task
