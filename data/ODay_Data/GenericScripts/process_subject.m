%% Example Script to demonstrate the Marker and IMU tracking Workflow.
% Master script and notes for Generic Subject Experiment
clear all; close all; clc;
import org.opensim.modeling.*
 

%% Define subject number and activity
subjectNumRaw = 3; % Change this to the desired subject number (e.g., 1, 2, ..., 11)
subjectNum = sprintf('%02d', subjectNumRaw); % Format as two digits (e.g., 1 becomes '01')
activity = 'walking'; % Change this to 'walking' or 'complexTasks'
endStamp = Inf; % Change this to the desired end time in seconds (e.g., 2, 5, etc.). Inf for no limit.

%% Section 1. Process Marker Based IK and Model Calibration
% Perform traditional marker based IK, with marker distances as the cost function,
% using the IKTool. 
% run_marker_ik(subjectNum, activity, -Inf, endStamp)

% Use the posed model from the marker based IK solution to calibrate the placement
% of the IMUs on the model.
calibrate_model(subjectNum, activity)

%% Section 2. Run IK for IMU Based Capture
% Perform IMU based IK, using the orientations of the marker plates as the cost
% function.
run_imu_ik(subjectNum, activity, 'mocap', -Inf, endStamp);

%% Section 3. Perform IMU Based IK
imuMethods = { 'Unprojected', 'Never Project', 'Mag Free'};
for methodIndex = 1:length(imuMethods)
    imuMethod = imuMethods{methodIndex};
    %% Section 3A. Align and Trim Marker and IMU Data
    if imuMethod == "xsens" || imuMethod == "mahony" || imuMethod == "madgwick"
        % create_trimmed_datasets(subjectNum, activity, imuMethod)
    end
    
    %% Section 3B. IMU Kinematics
    % Use the posed model and rotated IMU data to perform IK.
    run_imu_ik(subjectNum, activity, imuMethod, -Inf, endStamp)
end

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
    
    motionPath = sprintf('../Subject%s/%s/Mocap/ikResults/%s_IK.mot', subjectNum, activity, activity);
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

function create_trimmed_datasets(subjectNum, activity, imuMethod)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Aligning Marker and IMU Data on %s %s %s.-----\n", subjectNum, activity, imuMethod); % Added newline
    % TrimFromFrame/imuTrimFromFrame data for all subjects:
    % Subject1
    TrimFromFrame = 1;
    imuTrimFromFrame = 1; % Default to 1, will be updated based on subject
    switch subjectNum
        case '01'
            if activity == "walking"
                imuTrimFromFrame = 42999;
            else
                imuTrimFromFrame = 45573;
            end
        case '02'
            if activity == "walking"
                imuTrimFromFrame = 68436;
            else
                imuTrimFromFrame = 47010;
            end
        case '03'
            if activity == "walking"
                imuTrimFromFrame = 83444;
            else
                imuTrimFromFrame = 69692;
            end
        case '04'
            if activity == "walking"
                imuTrimFromFrame = 2818;
            else
                imuTrimFromFrame = 14780;
            end
        case '05'
            imuTrimFromFrame = 3350;
        case '06'
            if activity == "walking"
                imuTrimFromFrame = 457;
            else
                imuTrimFromFrame = 26780;
            end
        case '07'
            TrimFromFrame = 605;
        case '08'
            imuTrimFromFrame = 1576;
        case '09'
            TrimFromFrame = 313;
        case '10'
            TrimFromFrame = 254;
        case '11'
            TrimFromFrame = 293;
        otherwise
            warning('No specific trimming defined for Subject %s. Using default (1/1).\n', subjectNum); % Added newline
    end

    %% IMU Data Trimming
    % Generate a trimmed .sto file from the pregenerated IMU data .txt
    % files.
    % Define some folder and file paths
    imuPath = sprintf('../Subject%s/%s/IMU/%s', subjectNum, activity, imuMethod);
    orientationsFileName = fullfile(imuPath, sprintf('%s_orientations.sto', activity));
    
    % Instantiate an XsensDataReader
    mappingsFile = fullfile(sprintf('../Subject%s/%s/IMU', subjectNum, activity), sprintf('myIMUMappings_%s.xml', activity));
    xsensSettings = XsensDataReaderSettings(mappingsFile);
    xsens = XsensDataReader(xsensSettings);
    
    % Get a table reference for the data
    imuLowerExtremityPath = fullfile(imuPath, 'LowerExtremity/');
    tables = xsens.read(imuLowerExtremityPath);
    
    % Get Orientation Data as quaternions
    quatTable = xsens.getOrientationsTable(tables);
    nRows_imu = quatTable.getNumRows();
    fprintf('Untrimmed IMU data length: %d frames.\n', nRows_imu);
   
    % Apply IMU trimming if necessary
    if imuTrimFromFrame > 1
        imuTimeToTrim = quatTable.getIndependentColumn().get(imuTrimFromFrame - 1); % Adjust for 0-based indexing
        quatTable.trimFrom(imuTimeToTrim);
        fprintf('IMU data trimmed from frame %d for Subject %s. New length: %d frames.\n', imuTrimFromFrame, subjectNum, quatTable.getNumRows()); % Changed %s to %d for numRows
    end
    
    % Reset the time to start from Zero. Makes it easier to sync in GUI.
    % Ensure there are at least two rows to calculate rate
    if quatTable.getNumRows() > 1
        mRate = round(1/( quatTable.getIndependentColumn().get(1) - quatTable.getIndependentColumn().get(0)));
    else
        mRate = 100; % Default rate if not enough data to calculate
        warning('Not enough IMU data rows to calculate frame rate. Using default %d Hz.\n', mRate);
    end

    for i = 0 : quatTable.getNumRows() - 1
        quatTable.getIndependentColumn().set(i, i*(1/mRate));
    end
    
    fprintf("Saving trimmed IMU data to %s...\n", orientationsFileName)
    STOFileAdapterQuaternion.write(quatTable,  orientationsFileName);

    %% Marker Data Trimming
    mocapPath = sprintf('../Subject%s/%s/Mocap', subjectNum, activity);
    trcFile = sprintf('%s.trc', activity); % This is the correct variable name
    
    trcData = TimeSeriesTableVec3(fullfile(mocapPath, trcFile)); 
    ikFile = fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK.mot'));
    ikData = TimeSeriesTable(ikFile);
    
    % Apply Marker trimming if necessary
    if TrimFromFrame > 1
        timeToTrim = ikData.getIndependentColumn().get(TrimFromFrame - 1); % Adjust for 0-based indexing
        % Trim the IK trial
        disp('Performing trim of motion file ...');
        ikData.trimFrom(timeToTrim);
        fprintf('Marker motion data trimmed from frame %d for Subject %s. New length: %d frames.\n', TrimFromFrame, subjectNum, ikData.getNumRows());
        % Trim the trc file
        disp('Performing trim of trc file ...');
        trcData.trimFrom(timeToTrim);
        fprintf('TRC data trimmed from frame %d for Subject %s. New length: %d frames.\n', TrimFromFrame, subjectNum, trcData.getNumRows());
    else
        fprintf('No marker data trimming applied for Subject %s.\n', subjectNum);
    end
    
    % Reset the time to start from Zero. Makes it easier to sync in GUI.
    if ikData.getNumRows() > 1
        mRate = round(1/( ikData.getIndependentColumn().get(1) - ikData.getIndependentColumn().get(0)));
    else
        mRate = 100; % Default rate if not enough data to calculate
        warning('Not enough Marker IK data rows to calculate frame rate. Using default %d Hz.\n', mRate);
    end

    for i = 0 : ikData.getNumRows() - 1
        ikData.getIndependentColumn().set(i, i*(1/mRate));
        trcData.getIndependentColumn().set(i, i*(1/mRate));
    end

    trimmedTRCFile =  fullfile(mocapPath, strrep(trcFile, '.trc', '_trimmed.trc'));
    fprintf("Saving trimmed mocap data to %s...\n", trimmedTRCFile); % Added newline
    TRCFileAdapter().write(trcData, trimmedTRCFile); % Corrected trcFile
    
    trimmedIKFile = fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK_trimmed.mot'));
    fprintf("Saving trimmed mocap data to %s...\n", trimmedIKFile); % Added newline
    STOFileAdapter().write(ikData, trimmedIKFile); % Corrected trcFile
    
    % Display output
    disp('Trimming Complete. %s files saved to %s. Mocap files saved to %s and %s', imuMethod, orientationsFileName, trimmedTRCFile, trimmedIKFile);
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
