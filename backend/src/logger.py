import os 
import logging
from datetime import datetime


log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_folder=os.path.join(os.getcwd(),"LOGS")
os.makedirs(log_folder,exist_ok=True)

log_file_path=os.path.join(log_folder,log_file)

logging.basicConfig(level=logging.INFO, 
        filename=log_file_path,
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)