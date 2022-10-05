# LOAD LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURE SNS 

sns.set_style('white')

# LOAD DATA

df = sns.load_dataset('diamonds')

# PIPE AND FILTERING

(df
 .filter(['carat', 'color'])
 .query('color == "E"')
 .head(3)
)

# SUMMARIZATION

(df
 .filter(regex='^c')
 .query('cut in ["Ideal", "Premium"]')
 .groupby(['cut', 'color', 'clarity'])
 .agg(['mean', 'size'])
 .sort_values(by=('carat', 'mean'), ascending=False)
 .head()
)