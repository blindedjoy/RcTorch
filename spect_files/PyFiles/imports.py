
"""
if __name__ != "__main__":
	__name__ = "__main__"
print(__name__)
"""


# import packages
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import pickle

# import sub-packages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.io import loadmat
from datetime import date
import pathlib
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

from scipy.interpolate import Rbf