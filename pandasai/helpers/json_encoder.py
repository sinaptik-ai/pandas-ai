import datetime
from json import JSONEncoder

import numpy as np
import pandas as pd


def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(
        obj,
        (
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, np.int64):
        return float(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]

    return None


class CustomJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return obj.isoformat()

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="split")

        if numpy_converted := convert_numpy_types(obj):
            return numpy_converted

        return super().default(obj)
