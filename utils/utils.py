import yaml
import logging
from datetime import datetime

def load_configs(file_path = None):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def initialize_logging(file_dir = None):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{file_dir}/log_{current_time}.log"

    logging.basicConfig(level=logging.DEBUG,  
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),  
                            logging.StreamHandler() 
                        ])
    logger = logging.getLogger(__name__)
    return logger