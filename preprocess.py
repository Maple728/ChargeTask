import pandas as pd
import os
import re
from datetime import datetime, timedelta

# ---------------------- Constants -----------------
DATA_FILE = 'data/charge_data'
RAW_DATA_PATH = 'data/raw_data'

RECORD_COLUMNS = ['timeStart', 'timeEnd', 'power']

IN_DATE_FORMAT = '%Y.%m.%d %H:%M:%S'
OUT_DATE_FORMAT = '%Y%m%d%H%M'

# ---------------------- Functions -----------------
def parse_line_from_raw(line):
    ''' Return a list contains expected columns'''
    COMMON_COLUMN_REG = '"%s":([^,]*)'

    record = []
    for col_name in RECORD_COLUMNS:
        try:
            value = re.findall(COMMON_COLUMN_REG % col_name, line)[0]
            record.append(value.replace('"', ''))
        except Exception as e:
            return record
        
    return record

def process_record(records):
    ''' Return a record type is DataFrame whose columns is RECORD_COLUMNS'''
    records = pd.DataFrame(data = records, columns = RECORD_COLUMNS)
    # filter null field
    for col_name in RECORD_COLUMNS:
        records = records[ records[col_name] != 'null']
        
    records[['timeStart', 'timeEnd']] = records[['timeStart', 'timeEnd']].applymap(lambda r : datetime.strptime(r, IN_DATE_FORMAT).strftime(OUT_DATE_FORMAT) )
    
    return records

def read_raw_data():
    ''' Return records processed from raw data'''
    records = []
    # read and parse records from raw data file
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    tmp = parse_line_from_raw(line)
                    if len(tmp) == len(RECORD_COLUMNS):
                        records.append(tmp)
    # process record
    records = process_record(records)

    return records

def read_records():
    ''' Return all records of DataFrame from processed data or raw data'''
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        records = read_raw_data()
        records.to_csv(DATA_FILE, index = False)
        return records

# return list of <year, month, day, hour, power>
def hourly_process_one_record(record):
    result = []
    
    start = datetime.strptime(str(int(record.timeStart)), OUT_DATE_FORMAT)

    end = datetime.strptime(str(int(record.timeEnd)), OUT_DATE_FORMAT)

    gap = end - start

    if start.year < 2017:
        return []
    
    if gap.seconds <= 0:
        return []
    
    power_per_sec = float(record.power) / gap.seconds

    while start < end:
        tmp = datetime(start.year, start.month, start.day, start.hour)
        tmp = tmp + timedelta(hours = 1)

        if tmp > end:
            tmp = end
        
        # calc power in this hour of tmp
        power = (tmp - start).seconds * power_per_sec
        result.append([start.year, start.month, start.day, start.hour, power])
        
        # loop
        if tmp >= end:
            break
        start = tmp
     
    return result

def hourly_process(records):
    result = []
    for i in records.index:
        result.extend(hourly_process_one_record(records.iloc[i]))

    result = pd.DataFrame(data = result, columns = ['year', 'month', 'day', 'hour', 'power'])

    result = result.groupby(['year', 'month', 'day', 'hour']).sum().reset_index()

    # filter invalid records
    MIN_RECORDS_IN_DAY = 20
    result = result.groupby(['year', 'month', 'day']).filter(lambda x : x['hour'].count() > MIN_RECORDS_IN_DAY)
    return result

def itera():
    for name, group in results.groupby(['year', 'month', 'day']):
        print(name)
        print(group)

records = read_records()

results = hourly_process(records)
