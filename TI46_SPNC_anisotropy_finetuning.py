import spnc_ml as ml
import numpy as np
import os
from spnc import spnc_anisotropy

# Spoken digit variables
speakers = ['f1', 'f2', 'f3', 'f4', 'f5'] # blank for all

# Save file
filename = "accuracy_vs_memory"

# Open the save file
if os.path.exists("../txt_files/"+filename+".txt"):
    print("Overwriting "+filename)
    os.remove("../txt_files/"+filename+".txt")
save_file = open("../txt_files/"+filename+".txt", "a")

# Net Parameters
Nvirt = 25
m0 = 1e-1
bias = True

N = 1

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
spn = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spn.gen_signal_fast_delayed_feedback

theta_list = np.logspace(-1,1,1)

acc_test_vs_memory = []

for theta in theta_list:
    
    acc_test = []
    
    params = {'theta': theta,'gamma' : 0,'delay_feedback' : 1,'Nvirt' : Nvirt}
    
    for i in range(N):

        print("Nvirt = "+str(Nvirt))
        print("m0 = "+str(m0))
        print(params)
        
        # DO IT
        accuracy = ml.spnc_spoken_digits(speakers,Nvirt,m0,bias,transform,params,nfft=2048,return_accuracy=True)
        
        acc_test.append(accuracy)
    
    acc_test_vs_memory.append(acc_test)
    
print(acc_test_vs_memory)
    
# Save the parameters
save_file.write(str(Nvirt)+"\n")
save_file.write(str(m0)+"\n")
#save_file.write(str(params['theta'])+"\n")
save_file.write(str(params['gamma'])+"\n")
save_file.write(str(len(theta_list))+"\n")
save_file.write(str(N)+"\n")
for theta in theta_list:
    save_file.write(str(theta)+"\n")
    
# Save the results
for k in range(len(theta_list)):
    for i in range(N):
        save_file.write(str(acc_test_vs_memory[k][i])+"\n")
    
save_file.close()