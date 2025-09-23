import numpy as np
import pandas as pd

def first_non_na(*args):
    default_value = pd.NA
    if len(args) > 0:
        default_value = args[0]

    for value in args:
        if not pd.isna(value):
            return value

    return default_value
