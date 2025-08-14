import os
import logging
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# print(log_file)
# Create folder path for logs
logs_folder = os.path.join(os.getcwd(), "logs")
# print("Logs Folder:", logs_folder)

os.makedirs(logs_folder,exist_ok=True)

log_file_path = os.path.join(logs_folder, log_file)
# print("Log File Path:", log_file_path)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")



