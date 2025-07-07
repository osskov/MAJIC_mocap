%% Example Script to demonstrate the Marker and IMU tracking Workflow.
% Master script and notes for Generic Subject Experiment
clear all; close all;clc;

%% Define subject number and activity
subjectNumRaw = 3; % Change this to the desired subject number (e.g., 1, 2, ..., 11)
subjectNum = sprintf('%02d', subjectNumRaw); % Format as two digits (e.g., 1 becomes '01')
activity = 'walking'; % Change this to 'walking' or 'complexTasks'
endStamp = 60;

%% Section 1. Process Marker Based IK
% Perform Marker based IK using the IKTool. Tracking is performed using
% only the IMU plate based markers.
% run_marker_ik(subjectNum, activity, -Inf, endStamp)
   
%% Section 2. Create Calibrated Model
% calibrate_model(subjectNum, activity)

%% Run IK for IMU Based Capture
% run_imu_ik(subjectNum, activity, 'mocap', -Inf, endStamp);

imuMethods = {'majic'};%'madgwick', 'mahony', 'xsens'}; 
for methodIndex = 1:length(imuMethods)
    imuMethod = imuMethods{methodIndex};
    %% Section 3. Align and Trim Marker and IMU Data
    % create_trimmed_datasets(subjectNum, activity, imuMethod)
    
    %% Section 4. IMU Kinematics
    % Use the posed model and rotated IMU data to perform IK.
    run_imu_ik(subjectNum, activity, imuMethod, -Inf, endStamp)
end

%% Function Definitions
function run_marker_ik(subjectNum, activity, startTime, endTime)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Running MoCap Inverse Kinematics on %s %s.-----\n", subjectNum, activity); % Added newline
    ik = InverseKinematicsTool('Setup_IK_Walking.xml');
    
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
    
    disp("Launching IK tool...")
    % Run IK
    ik.run();
    disp("IK Complete.")
    
    model = Model(modelFile); 
    
    % Load the results of the IK run
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
    STOFileAdapter().write(markerMotion, fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK.mot')));
    fprintf("Updated IK motion file saved to %s\n", fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK.mot'))); % Added newline
end

function calibrate_model(subjectNum, activity)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Calibrating IMU Model on %s %s %s.-----\n", subjectNum, activity); % Added newline
    %% Generate a posed Model from the Marker Data.
    % The model is posed using the estimated kinematics from the (now synced)
    % marker based IK solution.
    modelFile = sprintf('../Subject%s/model_Rajagopal2015_registered.osim', subjectNum);
    modelFile_posed = sprintf('../Subject%s/model_Rajagopal2015_posed.osim', subjectNum);
    model = Model(modelFile);
    
    motionPath = sprintf('../Subject%s/%s/Mocap/ikResults/%s_IK.mot', subjectNum, activity, activity);
    markerMotion = TimeSeriesTable(motionPath);
    
    % Cycle through the model coordinates and update the default values.
    cs = model.getCoordinateSet();
    pelvis_rotation = 0;
    for i = 0 : cs.getSize() - 1
        % Get the Coordinate value from the the motion data.
        % Ensure the column index matches the coordinate index
        value = markerMotion.getRowAtIndex(0).getElt(0,i); % getElt(rowIndex, colIndex)

        % Check if the coordinate is Rotational and convert to radians
        if strcmp('Rotational', char(cs.get(i).getMotionType()))
            % Assuming pelvis_rotation corresponds to a specific coordinate, e.g., index 2
            if i == 2 % This index (2) needs to correspond to the pelvis rotation coordinate
                pelvis_rotation = value;
            end
            value = deg2rad(value);
        end
        cs.get(i).set_default_value( value );
    end
    % Print model to file.
    model.initSystem();
    model.print(modelFile_posed);
    
    %% Calibrate Posed Model
    baseIMUName = 'pelvis_imu'; visualizeCalibration = false;
    % Use the posed model to generate a IMU calibrated model.
    imu = IMUPlacer();
    imu.set_model_file(modelFile_posed);
   
    % orientationsFileName = fullfile(imuPath, sprintf('%s_orientations.sto', activity));
    orientationsFileName = sprintf('../Subject%s/%s/Mocap/%s_orientations.sto', subjectNum, activity, activity);
    imu.set_orientation_file_for_calibration(orientationsFileName);
    imu.set_base_heading_axis('z');
    imu.set_base_imu_label(baseIMUName);
    
    % Run the ModelCalibrator()
    imu.run(visualizeCalibration);
    
    % Write the Calibrated Model to file
    calibratedModel = imu.getCalibratedModel();
    modelFile_calibrated = sprintf('../Subject%s/model_Rajagopal2015_calibrated.osim', subjectNum);
    calibratedModel.print(modelFile_calibrated);
    disp("Model Calibration Complete.")
end

function create_trimmed_datasets(subjectNum, activity, imuMethod)
    % IMPORTANT: Import OpenSim modeling package within the function scope
    import org.opensim.modeling.*

    fprintf("-----Aligning Marker and IMU Data on %s %s %s.-----\n", subjectNum, activity, imuMethod); % Added newline
    % TrimFromFrame/imuTrimFromFrame data for all subjects:
    % Subject 1: 1/42896, Subject 2: 1/68233;, Subject 3: 1/83342,  Subject 4: 2/2616,
    % Subject 5: 1/3350,  Subject 6: 1/457,    Subject 7: 605/1,    Subject 8: 1/1576,
    % Subject 9: 313/1,  Subject 10: 254/1,   Subject 11:  293/1.
    TrimFromFrame = 1;
    imuTrimFromFrame = 1; % Default to 1, will be updated based on subject
    switch subjectNum
        case '01'
            imuTrimFromFrame = 42896;
        case '02'
            imuTrimFromFrame = 68233;
        case '03'
            imuTrimFromFrame = 83440;
        case '04'
            TrimFromFrame = 2;
            imuTrimFromFrame = 2616;
        case '05'
            imuTrimFromFrame = 3350;
        case '06'
            imuTrimFromFrame = 457;
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
    
    fprintf("Saving trimmed mocap data to %s...\n", fullfile(mocapPath, strrep(trcFile, '.trc', '_trimmed.trc'))); % Added newline
    TRCFileAdapter().write(trcData, fullfile(mocapPath, strrep(trcFile, '.trc', '_trimmed.trc'))); % Corrected trcFile
    
    fprintf("Saving trimmed mocap data to %s...\n", fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK_trimmed.mot'))); % Added newline
    STOFileAdapter().write(ikData, fullfile(mocapPath, 'ikResults', strrep(trcFile, '.trc', '_IK_trimmed.mot'))); % Corrected trcFile
    
    % Display output
    disp('Trimming Complete.');
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
   
    modelFile_calibrated = sprintf('../Subject%s/model_Rajagopal2015_calibrated.osim', subjectNum);
    
    resultsDirectory = fullfile(basePath, 'IKResults', sprintf('IKWithErrorsUniformWeights'));
    if ~exist(resultsDirectory, 'dir')
        mkdir(resultsDirectory);
    end
    
    % Determine the IK setup XML filename based on the activity
    % These are typically input files, not generated, so no prefix here.
    ikSetupXML = 'Setup_IK_Walking_IMU_extremely_low_feet_weights.xml';
    
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
    ik.set_output_motion_file(fullfile(resultsDirectory, sprintf("%s_IK.mot", activity)))
    
    disp("Launching IMU IK tool...")
    % Run IK
    ik.run();
    disp("IMU IK Complete.")
end
