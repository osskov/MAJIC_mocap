from toolchest.PlateTrial import PlateTrial
import os
for i in range(11):
    subject = f"Subject{i+1:02d}"
    for activity in ["walking", "complexTasks"]:
        # Question 1: Does the madgwick uniform weights IK exist?
        madgwick_ik_file = f"data/DO_NOT_MODIFY_AlBorno/{subject}/{activity}/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/{activity}_IK.mot"
        if os.path.exists(madgwick_ik_file):
            continue
        else:
            madgwick_ik_file = f"data/ODay_Data/{subject}/{activity}/IMU/madgwick/IKResults/IKUniformWeights/{activity}_IK.mot"
            if os.path.exists(madgwick_ik_file):
                continue
            else:
                print(f"Missing madgwick IK for {subject} - {activity}")

for i in range(11):
    subject = f"Subject{i+1:02d}"
    for activity in ["walking", "complexTasks"]:
        # Question 2: Do we have the original model osim file?
        original_model_osim = f"data/DO_NOT_MODIFY_AlBorno/{subject}/{activity}/IMU/madgwick/model_Rajagopal2015_calibrated.osim"
        if os.path.exists(original_model_osim):
            continue
        else:
            print(f"Missing original model osim for {subject} - {activity}")