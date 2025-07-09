%% Example Script to demonstrate the Marker and IMU tracking Workflow.
% Master script and notes for Generic Subject Experiment
clear all; close all; clc;
import org.opensim.modeling.*

subjects = {"01","02","03","04","05","06","07","08","09","10","11"};
activities = {'walking', 'complexTasks'};
methods = {"mocap", "unprojected", "never project", "mag free"};
endStamp = Inf; % Change this to the desired end time in seconds (e.g., 2, 5, etc.). Inf for no limit.

%% Section 0. Run Based IK if necessary
% Perform traditional marker based IK, with marker distances as the cost function,
% using the IKTool. Some trials from the original Al Borno dataset had to be reprocessed.
% The output of this function will then need to be trimmed using the
% PlateTrial found in the python code.
subjectNumNeedingIK = '01';
activityNeedingIK = 'complexTasks';
run_marker_ik(subjectNumNeedingIK, activityNeedingIK, -Inf, endStamp)

% for i = 1:length(subjects)
%     subject = subjects{i};
% 
%     for j = 1:length(activities)
%         activity = activities{j};
%         fprintf("~~~~~Processing data for Subject%s %s.~~~~~", subject, activity)
% 
%         %% Section 1. Calibrate the Registered Model
%         % Use the posed model from the marker based IK solution to calibrate the placement
%         % of the IMUs on     the model.
%         calibrate_model(subjectNum, activity)
% 
%         %% Section 2. Run IK for IMU Based Capture
%         % Perform IMU based IK, using the orientations of the marker plates as the cost
%         % function.
%         run_imu_ik(subjectNum, activity, 'mocap', -Inf, endStamp);
% 
%         %% Section 3. Perform IMU Based IK
%         for k = 1:length(methods)
%             method = methods{k};
%             run_imu_ik(subjectNum, activity, method, -Inf, endStamp)
%         end
%     end
% end

%% Function Definitions
function run_marker_ik(subjectNum, activity, startTime, endTime)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Running MoCap Inverse Kinematics on %s %s.-----\n", subjectNum, activity); % Added newline
    ik = InverseKinematicsTool('Setup_Marker_IK.xml');
    
    % This code uses the original registered model from the OpenSense paper
    modelFile = sprintf('../Subject%s/model_Rajagopal2015_registered.osim', subjectNum);
    ik.set_model_file(modelFile);

    % This code solves IK off of the original, untrimmed .trc files to make
    % sure the files are in sync
    mocapPath = sprintf('../Subject%s/%s/Mocap', subjectNum, activity);
    trcFile = sprintf('%s.trc', activity);
    ik.set_marker_file(fullfile(mocapPath,trcFile));
    
    % All results are stored in the ikResults path
    outputMotionFile = fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK.mot'));
    ik.set_output_motion_file(outputMotionFile);
    ik.set_results_directory(fullfile(mocapPath, 'ikResults'));
    
    % Set time range. Note: set_time_range takes two arguments (index, time).
    % Setting both 0 and 1 with -Inf and Inf means start from beginning, end at end.
    ik.set_time_range(0, startTime); 
    ik.set_time_range(1, endTime);
    
    disp("Launching IK tool...\n")
    % Run IK
    ik.run();
    disp("IK Complete.\n")
    
    model = Model(modelFile); 
    
    % Set the translation coordinated to the default values.
    markerMotion = TimeSeriesTable(outputMotionFile);
    labels = markerMotion.getColumnLabels();
    for i = 0 : markerMotion.getNumColumns() - 1
        if ~strcmp('Rotational', char(model.getCoordinateSet.get(labels.get(i)).getMotionType()))
            coord = model.getCoordinateSet.get(labels.get(i));
            defaultValue = coord.get_default_value();
            for u = 0 : markerMotion.getNumRows() - 1
                markerMotion.getRowAtIndex(u).set(i,defaultValue);
            end
        end
    end
    
    % Write model to file. This is overwriting an input file, not a new generated one, so no prefix.
    STOFileAdapter().write(markerMotion, outputMotionFile);
    fprintf("Updated IK motion file saved to %s\n", outputMotionFile); % Added newline
end

function calibrate_model(subjectNum, activity)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Calibrating IMU Model on Subject%s %s.-----\n", subjectNum, activity); % Added newline
    %% Generate a posed Model from the Marker Data.
    % The model is posed using the estimated kinematics from the (now synced)
    % marker based IK solution.
    modelFile = sprintf('../Subject%s/model_Rajagopal2015_registered.osim', subjectNum);
    modelFile_posed = sprintf('../Subject%s/model_Rajagopal2015_posed_%s.osim', subjectNum, activity);
    model = Model(modelFile);
    
    motionPath = sprintf('../Subject%s/%s/Mocap/ikResults/%s_IK_trimmed.mot', subjectNum, activity, activity);
    markerMotion = TimeSeriesTable(motionPath);
    
    % Cycle through the model coordinates and update the default values.
    cs = model.getCoordinateSet();
    for i = 0 : cs.getSize() - 1
        % Get the Coordinate value from the the motion data.
        % Ensure the column index matches the coordinate index
        value = markerMotion.getRowAtIndex(0).getElt(0,i); % getElt(rowIndex, colIndex)

        % Check if the coordinate is Rotational and convert to radians
        if strcmp('Rotational', char(cs.get(i).getMotionType()))
            % Assuming pelvis_rotation corresponds to a specific coordinate, e.g., index 2
            value = deg2rad(value);
        end
        cs.get(i).set_default_value( value );
    end
    % Print model to file.
    model.initSystem();
    model.print(modelFile_posed);
    
    %% Calibrate Posed Model
    visualizeCalibration = false;
    % Use the posed model to generate a IMU calibrated model.
    imu = IMUPlacer();
    imu.set_model_file(modelFile_posed);
   
    % orientationsFileName = fullfile(imuPath, sprintf('%s_orientations.sto', activity));
    orientationsFileName = sprintf('../Subject%s/%s/Mocap/%s_orientations.sto', subjectNum, activity, activity);
    imu.set_orientation_file_for_calibration(orientationsFileName);
    
    % Run the ModelCalibrator()
    imu.run(visualizeCalibration);
    
    % Write the Calibrated Model to file
    calibratedModel = imu.getCalibratedModel();
    modelFile_calibrated = sprintf('../Subject%s/model_Rajagopal2015_calibrated_%s.osim', subjectNum, activity);
    calibratedModel.print(modelFile_calibrated);
    fprintf("Model Calibration Complete. Model saved to %s.\n", modelFile_calibrated)
end

function run_imu_ik(subjectNum, activity, imuMethod, startTime, endTime)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Running IMU Inverse Kinematics on %s %s %s.-----\n", subjectNum, activity, imuMethod); % Added newline
    % Use the posed model and rotated IMU data to perform IK.
    if imuMethod == "mocap"
        basePath = sprintf('../Subject%s/%s/Mocap/', subjectNum, activity);
    else
        basePath = sprintf('../Subject%s/%s/IMU/%s', subjectNum, activity, imuMethod);
    end
   
    modelFile_calibrated = sprintf('../Subject%s/model_Rajagopal2015_calibrated_%s.osim', subjectNum, activity);

    resultsDirectory = fullfile(basePath, 'IKResults', sprintf('IKWithErrorsUniformWeights'));
    if ~exist(resultsDirectory, 'dir')
        mkdir(resultsDirectory);
    end
    
    % Determine the IK setup XML filename based on the activity
    % These are typically input files, not generated, so no prefix here.
    ikSetupXML = 'Setup_IMU_IK.xml';
    
    % Instantiate an InverseKinematicsStudy
    ik = IMUInverseKinematicsTool(ikSetupXML);
    
    % Set time range. Note: set_time_range takes two arguments (index, time).
    % -Inf and 10 is a valid range.
    ik.set_time_range(0, startTime); 
    ik.set_time_range(1, endTime);
    
    ik.set_model_file(modelFile_calibrated);
    orientationsFile = fullfile(basePath, sprintf('%s_orientations.sto', activity));
    ik.set_orientations_file(orientationsFile);
    ik.set_results_directory(resultsDirectory);
    outputMotionFile = fullfile(resultsDirectory, sprintf("%s_IK.mot", activity));
    ik.set_output_motion_file(outputMotionFile);
    if imuMethod == "xsens" || imuMethod == "mahony" || imuMethod == "madgwick"
        ik.set_sensor_to_opensim_rotations(Vec3(-pi/2,0,0));
    end
    disp("Launching IMU IK tool...")
    % Run IK
    ik.run();
    fprintf("IMU IK Complete. File saved to: %s\n", outputMotionFile); % Added newline
end
