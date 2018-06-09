import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CNN polling
cnn_pool = pd.read_csv('./cnn_pool.csv', header=0)

time = np.array(cnn_pool['Wall time'])[:930]
hit_rate = np.array(cnn_pool['Value'])[:930]

init_time = time[0]
real_time = [(t - init_time) / (60 * 60) for t in time]
real_hit_rate = [h * 100 for h in hit_rate]

plt.plot(real_time, real_hit_rate)


# CNN batchnorm
cnn_no_pool_bn_pt = pd.read_csv('./cnn_no_pool_bn_pt.csv', header=0)

time = np.array(cnn_no_pool_bn_pt['Wall time'])
hit_rate = np.array(cnn_no_pool_bn_pt['Value'])

init_time = time[0]
real_time = [(t - init_time) / (60 * 60) for t in time]
real_hit_rate = [h * 100 for h in hit_rate]

plt.plot(real_time, real_hit_rate)


plt.xlabel('time (hour)')
plt.ylabel('hit rate (%)')
plt.show()
