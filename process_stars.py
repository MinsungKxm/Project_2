import pandas as pd
import numpy as np

# Load raw dataset
df = pd.read_csv('./data/hygdata_v37.csv', low_memory=False)

# Remove the Sun
df = df[df['proper'] != 'Sol']

# Remove invalid distances
df['dist'].replace(to_replace=100000, value=np.nan, inplace=True)

# Keep only stars visible to human eye
df = df[df['mag'] <= 6.5]

# Save processed file
df.to_csv('./data/stars.csv', index=False)

print("Done! stars.csv created.")
