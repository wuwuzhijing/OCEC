
import os
OCEC_DATA_ROOT='/ssddisk/guochuang/ocec/'
OCEC_CODE_ROOT='/103/guochuang/Code/myOCEC/'
OCEC_LOG_DIR=OCEC_CODE_ROOT + 'logs'
TXT_SUFFIX='.txt'

CSV_DIR = OCEC_DATA_ROOT + 'list_hq'
BASE_DIR = os.path.basename(CSV_DIR)


REPORT_DIR = OCEC_LOG_DIR + '/dataset/' + BASE_DIR

print('REPORT_DIR = ', REPORT_DIR)

REPORT_FILENAME = 'dataset_stats_report_' + BASE_DIR + TXT_SUFFIX

print('REPORT_FILENAME = ', REPORT_FILENAME)