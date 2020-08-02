import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
case_data=pd.read_csv('Data/Main.csv')
x_data = np.array(case_data['serial'])
y_data = np.array(case_data['Total Confirmed'])

log_x_data = np.log(x_data)
log_y_data = np.log(y_data)

curve_fit = np.polyfit(x_data, log_y_data, 1)
#print(curve_fit)
y = np.exp(1.11292745) * np.exp(0.08338703*x_data)
plt.plot(x_data, y_data,color='red')
plt.plot(x_data, y,color='yellow')
plt.show()
