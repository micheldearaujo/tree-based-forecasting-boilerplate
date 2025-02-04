# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import os
import logging.config
import yaml
import locale
import datetime as dt

import pandas as pd
import pandas_gbq

# from src.config import *
from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *

TARGET_COL = model_config['target_col']
PREDICTED_COL = model_config['predicted_col']
FORECAST_HORIZON = model_config['forecast_horizon']
N_SPLITS = model_config['n_windows']

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)


class TableJoining:
    """
    How to call this:
        from src.data.features import TableJoining
        joiner = TableJoining()
        TableJoining.run_all()
    """
    
    def __init__(self):
        # self.save_to_table = save_to_table
        self.raw_data_path = "./data/raw/"
        self.interim_data_path = "./data/interim/"
        self.read_project_id = 'sz-vision-feat'
        self.write_project_id = 'sz-vision-feat'


    def join_ip_3m_tables(self, dataset, save_to_table=False):

        clean_pix_china = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_china",
                                   project_id=self.read_project_id)

        clean_bsis = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_baltic_exchange",
                                   project_id=self.read_project_id)

        clean_cpi = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_cpi",
                                   project_id=self.read_project_id)

        clean_pix_europe = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_europe",
                                   project_id=self.read_project_id)
        clean_moody = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_moody_index",
                                   project_id=self.read_project_id)

        clean_pmi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_pmi",
                                   project_id=self.read_project_id)

        clean_ppi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_ppi",
                                   project_id=self.read_project_id)

        clean_retail_sales = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_retail_sales",
                                   project_id=self.read_project_id)

        clean_shfe = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_shfe",
                                   project_id=self.read_project_id)

        ip_3m_df = (
            clean_pix_china
            .merge(clean_bsis, on = ['friday_week'], how = 'left')
            .merge(clean_cpi, on = ['friday_week'], how = 'left')
            .merge(clean_pix_europe, on = ['friday_week'], how = 'left')
            .merge(clean_moody, on = ['friday_week'], how = 'left')
            .merge(clean_pmi, on = ['friday_week'], how = 'left')
            .merge(clean_ppi, on = ['friday_week'], how = 'left')
            .merge(clean_retail_sales, on = ['friday_week'], how = 'left')
            .merge(clean_shfe, on = ['friday_week'], how = 'left')
        )
        ip_3m_df.sort_values('friday_week', inplace=True)

        if save_to_table:
            print(f'Writing the current IP3M features to Data Lake...')
            ip_3m_df.to_csv(f"./data/processed/ip_3m_weekly_{short_month_str}.csv", index=False)
            locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
            today_date = dt.datetime.today()
            short_month_str = today_date.strftime('%b')

            pandas_gbq.to_gbq(dataframe = ip_3m_df, destination_table=f"artefact.features_ip_3m_weekly_{short_month_str}", project_id=self.write_project_id, if_exists='replace')
            
        return ip_3m_df


    def join_ip_45d_tables(self, dataset, save_to_table=False):

        clean_pix_china = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_china",
                                   project_id=self.read_project_id)
        clean_backlog = pandas_gbq.read_gbq(f"SELECT * except (original_date) FROM {dataset}.interim_backlog",
                                   project_id=self.read_project_id)
        clean_bsis = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_baltic_exchange",
                                   project_id=self.read_project_id)
        clean_bhkp_woodchips = pandas_gbq.read_gbq(f"SELECT * except(original_date, date)  FROM {dataset}.interim_woodchips",
                                   project_id=self.read_project_id)
        clean_bcom = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_bcom",
                                   project_id=self.read_project_id)
        clean_fx = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_fx",
                                   project_id=self.read_project_id)
        clean_moody = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_moody_index",
                                   project_id=self.read_project_id)
        clean_pmi = pandas_gbq.read_gbq(f"SELECT *  except(original_date) FROM {dataset}.interim_pmi",
                                   project_id=self.read_project_id)
        clean_ppi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_ppi",
                                   project_id=self.read_project_id)
        clean_stocks = pandas_gbq.read_gbq(f"SELECT * except(original_date ) FROM {dataset}.interim_stocks",
                                   project_id=self.read_project_id)
        clean_oi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_order_intake",
                                   project_id=self.read_project_id)
        clean_tissue_stock = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_stock",
                                   project_id=self.read_project_id)
        clean_tissue_price = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_tissue_price",
                                   project_id=self.read_project_id)     
        clean_resale = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_resale",
                                   project_id=self.read_project_id)  
        
        ip_45d_df = (
            clean_pix_china
            .merge(clean_backlog, on = ['friday_week'], how = 'left')
            .merge(clean_bhkp_woodchips, on = ['friday_week'], how = 'left')
            .merge(clean_bcom, on = ['friday_week'], how = 'left')
            .merge(clean_bsis, on = ['friday_week'], how = 'left')
            .merge(clean_fx, on = ['friday_week'], how = 'left')
            .merge(clean_moody, on = ['friday_week'], how = 'left')
            .merge(clean_pmi, on = ['friday_week'], how = 'left')
            .merge(clean_ppi, on = ['friday_week'], how = 'left')
            .merge(clean_stocks, on = ['friday_week'], how = 'left')
            .merge(clean_oi, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_price, on = ['friday_week'], how = 'left')
            .merge(clean_resale, on = ['friday_week'], how = 'left')
        )
        ip_45d_df.sort_values('friday_week', inplace=True)

        if save_to_table:
            print(f'Writing the current IP45D features to Data Lake...')
            locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
            today_date = dt.datetime.today()
            short_month_str = today_date.strftime('%b')

            ip_45d_df.to_csv(f"./data/processed/ip_45d_weekly_{short_month_str}.csv", index=False)
            pandas_gbq.to_gbq(dataframe = ip_45d_df, destination_table=f"artefact.features_ip_45d_weekly_{short_month_str}", project_id=self.write_project_id, if_exists='replace')



    def join_all_tables(self, dataset, save_to_table=False):

        clean_pix_china = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_china",
                                   project_id=self.read_project_id)
        clean_cpi = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_cpi",
                                   project_id=self.read_project_id)
        clean_pix_europe = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_europe",
                                   project_id=self.read_project_id)
        clean_backlog = pandas_gbq.read_gbq(f"SELECT * except (original_date) FROM {dataset}.interim_backlog_weekly",
                                   project_id=self.read_project_id)
        clean_bsis = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_baltic_exchange",
                                   project_id=self.read_project_id)
        clean_bhkp_woodchips = pandas_gbq.read_gbq(f"SELECT * except(original_date, date)  FROM {dataset}.interim_woodchips",
                                   project_id=self.read_project_id)
        clean_bcom = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_bcom",
                                   project_id=self.read_project_id)
        clean_fx = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_fx",
                                   project_id=self.read_project_id)
        clean_moody = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_moody_index",
                                   project_id=self.read_project_id)
        clean_pmi = pandas_gbq.read_gbq(f"SELECT *  except(original_date) FROM {dataset}.interim_pmi",
                                   project_id=self.read_project_id)
        clean_ppi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_ppi",
                                   project_id=self.read_project_id)
        clean_stocks = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_stocks",
                                   project_id=self.read_project_id)
        clean_oi = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_order_intake_ip_model_weekly",
                                    project_id=self.read_project_id).sort_values('friday_week')
        clean_tissue_stock = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_stock",
                                   project_id=self.read_project_id)
        clean_tissue_sales = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_sales",
                                   project_id=self.read_project_id)
        clean_tissue_export = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_export",
                                   project_id=self.read_project_id)
        clean_retail_sales = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_retail_sales",
                                   project_id=self.read_project_id)
        clean_shfe = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_shfe",
                                   project_id=self.read_project_id)   
        clean_resale = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_resale",
                                   project_id=self.read_project_id)  
        clean_tissue_price = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_tissue_price",
                                   project_id=self.read_project_id)                       
        
        all_features_df = (
            clean_pix_china
            .merge(clean_pix_europe, on = ['friday_week'], how = 'left')
            .merge(clean_cpi, on = ['friday_week'], how = 'left')
            .merge(clean_backlog, on = ['friday_week'], how = 'left')
            .merge(clean_bhkp_woodchips, on = ['friday_week'], how = 'left')
            .merge(clean_bcom, on = ['friday_week'], how = 'left')
            .merge(clean_bsis, on = ['friday_week'], how = 'left')
            .merge(clean_fx, on = ['friday_week'], how = 'left')
            .merge(clean_moody, on = ['friday_week'], how = 'left')
            .merge(clean_pmi, on = ['friday_week'], how = 'left')
            .merge(clean_ppi, on = ['friday_week'], how = 'left')
            .merge(clean_stocks, on = ['friday_week'], how = 'left')
            .merge(clean_oi, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_stock, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_sales, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_export, on = ['friday_week'], how = 'left')
            .merge(clean_retail_sales, on = ['friday_week'], how = 'left')
            .merge(clean_shfe, on = ['friday_week'], how = 'left')
            .merge(clean_resale, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_price, on = ['friday_week'], how = 'left')
        )
        all_features_df.sort_values('friday_week', inplace=True)

        if save_to_table:
            print(f'Writing All Features DF to Data Lake...')

            locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
            today_date = dt.datetime.today()
            short_month_str = today_date.strftime('%b')

            all_features_df.to_csv(f"./data/processed/features_all_features_weekly_{short_month_str}.csv", index=False)
            pandas_gbq.to_gbq(dataframe = all_features_df, destination_table=f"artefact.features_all_features_weekly_{short_month_str}", project_id=self.write_project_id, if_exists='replace')

        return all_features_df

    def join_elasticity_monthly_tables(self, dataset, save_to_table=False):

        clean_pix_china_monthly = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_china_monthly",
                                   project_id=self.read_project_id)
        
        clean_bsis_monthly = pandas_gbq.read_gbq(f"SELECT *  FROM {dataset}.interim_baltic_exchange_monthly",
                                   project_id=self.read_project_id)
        clean_bhkp_woodchips_monthly = pandas_gbq.read_gbq(f"SELECT *  FROM {dataset}.interim_woodchips_monthly",
                                   project_id=self.read_project_id)
        clean_bcom_monthly = pandas_gbq.read_gbq(f"SELECT *  FROM {dataset}.interim_bcom_monthly",
                                   project_id=self.read_project_id)
        clean_stocks_monthly = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_stocks_monthly",
                                   project_id=self.read_project_id)
        clean_resale_monthly = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_resale_monthly",
                                   project_id=self.read_project_id) 
        clean_order_intake_monthly = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_order_intake_monthly",
                                   project_id=self.read_project_id) 

        
        elasticity_df = (
            clean_order_intake_monthly
            .merge(clean_pix_china_monthly, on = ['end_of_month'], how = 'left')
            .merge(clean_bsis_monthly, on = ['end_of_month'], how = 'left')
            .merge(clean_bhkp_woodchips_monthly, on = ['end_of_month'], how = 'left')
            .merge(clean_bcom_monthly, on = ['end_of_month'], how = 'left')
            .merge(clean_stocks_monthly, on = ['end_of_month'], how = 'left')
            .merge(clean_resale_monthly, on = ['end_of_month'], how = 'left')
        )
        elasticity_df.sort_values('end_of_month', inplace=True)

        if save_to_table:
            print(f'Writing Elasticity DF to Data Lake...')
            pandas_gbq.to_gbq(dataframe = elasticity_df, destination_table='artefact.elasticity_monthly_out', project_id=self.write_project_id, if_exists='replace')

    def join_elasticity_weekly_tables(self, dataset, save_to_table=False):

        clean_pix_china = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_china",
                                   project_id=self.read_project_id)
        clean_cpi = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_cpi",
                                   project_id=self.read_project_id)
        clean_pix_europe = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_foex_pulp_europe",
                                   project_id=self.read_project_id)
        # clean_backlog = pandas_gbq.read_gbq(f"SELECT * except (original_date) FROM {dataset}.interim_backlog_weekly",
        #                            project_id=self.read_project_id)
        clean_bsis = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_baltic_exchange",
                                   project_id=self.read_project_id)
        clean_bhkp_woodchips = pandas_gbq.read_gbq(f"SELECT * except(original_date)  FROM {dataset}.interim_woodchips",
                                   project_id=self.read_project_id)
        clean_bcom = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_bcom",
                                   project_id=self.read_project_id)
        # clean_fx = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_fx",
        #                            project_id=self.read_project_id)
        clean_moody = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_moody_index",
                                   project_id=self.read_project_id)
        clean_pmi = pandas_gbq.read_gbq(f"SELECT *  except(original_date) FROM {dataset}.interim_pmi",
                                   project_id=self.read_project_id)
        clean_ppi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_ppi",
                                   project_id=self.read_project_id)
        clean_stocks = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_stocks",
                                   project_id=self.read_project_id)
        # clean_oi = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_order_intake",
                                #    project_id=self.read_project_id)
        # clean_oi = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_order_intake_ip_model_weekly",
        #                             project_id=self.read_project_id).sort_values('friday_week')
        clean_tissue_stock = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_stock",
                                   project_id=self.read_project_id)
        clean_tissue_sales = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_sales",
                                   project_id=self.read_project_id)
        clean_tissue_export = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_tissue_export",
                                   project_id=self.read_project_id)
        clean_retail_sales = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_retail_sales",
                                   project_id=self.read_project_id)
        # clean_shfe = pandas_gbq.read_gbq(f"SELECT * except(original_date) FROM {dataset}.interim_shfe",
        #                            project_id=self.read_project_id)   
        clean_resale = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_resale",
                                   project_id=self.read_project_id)  
        clean_tissue_price = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_tissue_price",
                                   project_id=self.read_project_id)
        
        clean_order_intake_weekly = pandas_gbq.read_gbq(f"SELECT * FROM {dataset}.interim_order_intake_weekly",
                                   project_id=self.read_project_id)
        
        elasticity_weekly_df = (
            clean_pix_china
            .merge(clean_order_intake_weekly, on = ['friday_week'], how = 'left')
            .merge(clean_pix_europe, on = ['friday_week'], how = 'left')
            .merge(clean_cpi, on = ['friday_week'], how = 'left')
            .merge(clean_bhkp_woodchips, on = ['friday_week'], how = 'left')
            .merge(clean_bcom, on = ['friday_week'], how = 'left')
            .merge(clean_bsis, on = ['friday_week'], how = 'left')
            .merge(clean_moody, on = ['friday_week'], how = 'left')
            .merge(clean_pmi, on = ['friday_week'], how = 'left')
            .merge(clean_ppi, on = ['friday_week'], how = 'left')
            .merge(clean_stocks, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_stock, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_sales, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_export, on = ['friday_week'], how = 'left')
            .merge(clean_retail_sales, on = ['friday_week'], how = 'left')
            .merge(clean_resale, on = ['friday_week'], how = 'left')
            .merge(clean_tissue_price, on = ['friday_week'], how = 'left')
        )
        elasticity_weekly_df.sort_values('friday_week', inplace=True)

        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
        today_date = dt.datetime.today()
        short_month_str = today_date.strftime('%b')

        if save_to_table:
            print(f'Writing Elasticity DF to Data Lake...')
            pandas_gbq.to_gbq(dataframe = elasticity_weekly_df, destination_table=f'artefact.elasticity_weekly_{short_month_str}', project_id=self.write_project_id, if_exists='replace')

    @classmethod
    def run_all(cls):
        instance = cls()
        save_to_table = True

        ######## Here you will find how to call the preprocessing for each one of the
        ######## Available databases, although the majority of them will be commented
        ######## since only a small portion is frequently used throughout the project.
        ######## To run the Production version of the project, refer to apps/01-preprocessing_and_join.


        # ip_3m_df = instance.join_ip_3m_tables(dataset = 'artefact', save_to_table=save_to_table)
        ip_45d_df = instance.join_ip_45d_tables(dataset = 'artefact', save_to_table=save_to_table)
        all_features_df = instance.join_all_tables(dataset = 'artefact', save_to_table=save_to_table)
        # elasticity_monthly_df = instance.join_elasticity_monthly_tables(dataset = 'artefact', save_to_table=save_to_table)
        # elasticity_weekly_df = instance.join_elasticity_weekly_tables(dataset = 'artefact', save_to_table=save_to_table)

if __name__ == '__main__':
    TableJoining().run_all()