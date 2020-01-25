import os
# initialize image dimensions 
IMAGE_SIZE = 224
BATCH_SIZE = 5 #5
EPOCH = 60

# initialize Redis Connection Settings
REDIS_HOST = "127.0.0.1"
REDIS_port = 6379
REDIS_DB = 0

# directory path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_ROOT = os.path.dirname(APP_ROOT)
DATASET_ROOT = os.path.join(SERVER_ROOT, 'database/dataSet')
DATAMODEL_ROOT = os.path.join(SERVER_ROOT, 'database/dataModel')
DATAUPLOAD_ROOT = os.path.join(SERVER_ROOT,'database/dataUpload')
DATATEMP_ROOT = os.path.join(SERVER_ROOT,'database/dataTemp')
