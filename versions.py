import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yfin

print("NumPy: " + np.__version__)
print("Pandas: " + pd.__version__)
print("Streamlit: " + st.__version__)
print("Yahoo Finance: " + yfin.__version__)

# NumPy: 1.24.2
# Pandas: 2.0.0
# Streamlit: 1.23.1
# Yahoo Finance: 0.2.20