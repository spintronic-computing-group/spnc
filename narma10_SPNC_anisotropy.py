# Run narma10 for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_anisotropy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 0.003
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 20
params = {'theta': 0.3,'gamma' : .113,'delay_feedback' : 0,'Nvirt' : Nvirt}
spn = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test,y_pred)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# visualize the results

plt.plot(y_test[100:200], label='Desired Output')  # 统一使用 y_test 作为目标输出
plt.plot(y_pred[100:200], label='Model Output')   # 统一使用 y_pred 作为模型预测输出
plt.legend(loc='lower left')
plt.xlabel('Time')
plt.ylabel('NARMA10 Output')
plt.title('NARMA10 Model vs Desired Output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_test[:], y_pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()


# y_test_df = pd.DataFrame(y_test)
# y_pred_df = pd.DataFrame(y_pred)

# y_test_df.to_csv('../csv_files/y_test.csv')
# y_pred_df.to_csv('../csv_files/y_pred.csv')