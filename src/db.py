# module db

# system
from datetime import datetime, timedelta

# db
from pymongo.collection import Collection

# processing
import pandas as pd


def db_log_insert(col: Collection, face_id: str, status: str, info: dict = None, export_csv: str = None):
    """Adds new log entry to DB collection

    Args:
        col (Collection): collection object
        face_id (str): face id
        status (str): new or found
        info (dict): additional information. Defaults to None.
        export_csv (str): export csv filename. Defaults to None.

    Note:
        export_csv: if None do nothing
    """
    # prepare
    query = {
        'time': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'status': status,
        'id': face_id,
    }

    # add other info if neccessary
    if info: 
        query.update(info)

    # insert into db
    col.insert_one(query)

    # export to csv
    if export_csv:
        docs = col.find()
        df = pd.DataFrame(docs)
        df.pop('_id')
        df.to_csv('db.csv', index=False)