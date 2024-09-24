import numpy as np
import pandas as pd

def loadfile(file):
    data = pd.read_csv(file)
    return data

