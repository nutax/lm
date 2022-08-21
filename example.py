import numpy as np
import pandas as pd
from lm import lm

dataset = pd.read_csv('sample.csv')
xx = (dataset[['Year', "Population"]]).to_numpy()
y = (dataset[['Employed']]).to_numpy()

print(lm(xx, y))