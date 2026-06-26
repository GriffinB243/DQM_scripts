import os
import numpy as np
import time

outfolder_r0 = '/data/user/fbivens5020/mock_data/' #THIS SHOULD BE ONLY A FOLDER UNDER YOUR USER
outfolder_npys = outfolder_r0 #THIS SHOULD BE ONLY A FOLDER UNDER YOUR USER
time_per_subrun = 50 # IN SECONDS
run = 400196
end_subrun = 15

os.system(f'rm -r {outfolder_r0}*')
os.system(f'rm -r {outfolder_npys}*')

r0_filename = '/data/wipac/CTA/targetcdata/run{0}_subrun{1}_r0.tio'

fee_temp_file = outfolder_npys + 'temperatures_FEEs_run{0}_subrun{1}.npy'
fpm_temp_file = outfolder_npys + 'temperatures_FPMs_run{0}_subrun{1}.npy'
fpm_hv_file = outfolder_npys + 'HV_FPMs_run{0}_subrun{1}.npy'
fee_current_file = outfolder_npys + 'Current_FEEs_run{0}_subrun{1}.npy'

def generate_temps_FEEs(run, subrun):
    temps = np.random.normal(35, 3, (22*2))
    try:
        np.save(fee_temp_file.format(run, subrun), temps)
    except:
        pass

def generate_temps_FPMs(run, subrun):
    temps = np.random.normal(30, 3, (22*4))
    try:
        np.save(fpm_temp_file.format(run, subrun), temps)
    except:
        pass

def generate_HV_current(run, subrun):
    hv = np.random.normal(33, 1, (22*1))
    current = np.random.normal(0.1, 0.02, (22*1))
    try:
        np.save(fpm_hv_file.format(run, subrun), hv)
    except:
        pass
    
    try:
        np.save(fee_current_file.format(run, subrun), current)
    except:
        pass

    
for subrun in range(0, end_subrun+1):
    time.sleep(time_per_subrun)
    print(f"Generating... Run: {run}, Subrun: {subrun}")
    os.system(f'cp {r0_filename.format(run, subrun)} {outfolder_r0}')
    generate_temps_FPMs(run, subrun)
    generate_temps_FEEs(run, subrun)
    generate_HV_current(run, subrun)

print("END OF SIMULATION")
