import datetime
from typing import List


def run_timestamp_preprocess(values: List[str]) -> List[str]:
    timestamp = values[2]

    date = datetime.datetime.strptime(timestamp, "%y%m%d%H")
    hour = str(date.hour)
    weekday = str(date.weekday())
    is_weekend = str(date.weekday() in [5, 6])

    return [hour, weekday, is_weekend]
