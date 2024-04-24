# module db

# system
from datetime import datetime, timedelta

# db
from pymongo.collection import Collection

# processing
import pandas as pd


def db_log_insert(col: Collection, face_id: str, status: str, export_csv: str = None):
    """Adds new log entry to DB collection

    Args:
        col (Collection): collection object
        face_id (str): face id
        status (str): new or found
        export_csv (str): export csv filename. Defaults to None.

    Note:
        export_csv: if None do nothing
    """
    col.insert_one({
        'time': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'status': status,
        'id': face_id,
    })

    # export to csv
    if export_csv:
        docs = col.find()
        df = pd.DataFrame(docs)
        df.pop('_id')
        df.to_csv('db.csv', index=False)