import pandas as pd
import numpy as np
import pandas_ta as ta

# Mock data
df = pd.DataFrame({
    "Close": np.random.uniform(100, 200, 50),
    "High":  np.random.uniform(100, 200, 50),
    "Low":   np.random.uniform(100, 200, 50),
    "Volume": np.random.randint(1000000, 5000000, 50),
})

print("=== bbands columns ===")
bb = ta.bbands(df["Close"], length=20)
print(bb.columns.tolist())

print("\n=== macd columns ===")
macd = ta.macd(df["Close"])
print(macd.columns.tolist())

print("\n=== stoch columns ===")
stoch = ta.stoch(df["High"], df["Low"], df["Close"])
print(stoch.columns.tolist())

print("\n=== adx columns ===")
adx = ta.adx(df["High"], df["Low"], df["Close"])
print(adx.columns.tolist())

print("\nDone!")
