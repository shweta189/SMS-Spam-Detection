import pandas as pd 
import os 
from pathlib import Path
from sklearn.model_selection import train_test_split 
import logging
# import yaml 

# checking if the logging directory's existence
# log_dir = 'logs'
# os.makedirs(log_dir,exist_ok=True)

#logging configuration

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# log_file_path = os.path.join(log_dir,'data_ingestion.log')
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# def load_params(params_path: str) -> dict:
#     """Load Parameters from YAML files."""
#     try:
#         with open(params_path,'r') as file:
#             params = yaml.safe_load(params_path)
#         logger.debug('Parameters retrieved from %s',params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('File not found: %s',params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('YAML error: %s',e)
#         raise
#     except Exception as e:
#         logger.error('YAML error: %s',e)
#         raise

def load_data(data_url:str) ->pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(data_url,encoding='latin-1')
        logger.debug("Data loaded from %s",data_url)
        return df 
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s',e)
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
        df.columns = ['target','text']
        df = df.drop_duplicates(keep='first')
        return df 
    except KeyError as e:
        logger.error("Missing column in DataFrame: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred during preprocess: %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str) ->pd.DataFrame:
    """Save the train and test datasets"""
    try:
        raw_data_path = os.path.join(data_path,'raw_data')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'))
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'))
        logger.debug("train and test data saved to %s",raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s',e)
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, ".."))
        test_size=0.2
        csv_path = os.path.join(root_dir, "experiment", "spam.csv")
        data_path = os.path.join(root_dir, "data") 
        df = load_data(csv_path)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path = data_path)
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s',e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()


