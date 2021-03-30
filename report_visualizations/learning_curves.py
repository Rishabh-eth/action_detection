import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# read data from result csv files
mAP = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v16/run-tensorboard-tag-Val_mAP.csv')
mAP = mAP.values
#mAP = mAP[:,2]

loss = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v16/run-tensorboard-tag-Train_loss.csv')
loss = loss.values
#loss = loss[:,2]

# more sophisticated loss sampling method
first = loss[0,1]
last = loss[-1,1]
loss[:,1] = loss[:,1] - first
loss[:,1] = loss[:,1] / (last - first) * 14
loss_final = loss[:,2]
batch = loss[:,1]
# find the loss array with an average value per epoch
"""
batches_per_epoch = int(loss.shape[0] / mAP.shape[0])
loss_final = np.zeros((mAP.shape[0],))

for i in range(loss_final.shape[0]):
    print(loss[i*batches_per_epoch:(i+1)*batches_per_epoch,2].shape)
    average = np.mean(loss[i*batches_per_epoch:(i+1)*batches_per_epoch,2])
    loss_final[i] = average
"""




mAP_final = mAP[:,2] * 100

# Create some mock data
epoch = np.arange(1, 15, 1)
fig, ax1 = plt.subplots(figsize=(7,3.5))

ax1.set_xlabel('epoch')
ax1.set_ylabel('Binary Cross Entropy (epoch average)')
ax1.set_title('Learning curves AVA-5k')
lns1 = ax1.plot(batch, loss_final, color='lightskyblue', label='Binary Cross Entropy Loss (Training)')
ax1.tick_params(axis='y')
ax1.grid()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('mAP(%)')  # we already handled the x-label with ax1
plt.ylim(ymax=47, ymin=37)
lns2 = ax2.plot(epoch, mAP_final, marker='o', color='#4267B2', label='mAP (Validation)')
ax2.tick_params(axis='y')
ax2.grid()


lns = lns1 + lns2
labs = [l.get_label() for l in lns]
l = ax2.legend(lns, labs, loc=0)
l.set_zorder(100)



fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('/home/sieberl/SA2020/pyslowfast/report_visualizations/learningcurves.jpg', bbox_inches='tight')
