
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_pickle('data/train')
valid = pd.read_pickle('data/valid')
print(train)
print(valid)
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['BTC-USD'], lw=1)
plt.plot(valid[['BTC-USD', 'Predictions']], lw=1)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()