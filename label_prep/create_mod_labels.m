clc;
clear;
close all;

%%
% Replace root_dir with actual path to KITTI_DATASET_ROOT
root_dir = '/path/to/kitti/dataset/root';

% get sub-directories
label_dir = fullfile(root_dir, 'raw', 'data_object_label_2', 'training', 'label_2');
calib_dir = fullfile(root_dir, 'raw', 'data_object_calib', 'training', 'calib');

label_mod_dir = fullfile(root_dir, 'raw', 'data_object_mod_label_2' , 'training', 'label_2');
if exist(label_mod_dir, 'dir') == 7
    rmdir(label_mod_dir);
end
mkdir(label_mod_dir);

files = dir(fullfile(label_dir, '*.txt'));
files = {files(:).name}';

%%
for i = 1:length(files)
    f = fopen(fullfile(label_mod_dir, files{i}), 'w');
    
    % load projection matrix
    P = readCalibration(calib_dir, i-1, 2);
    
    % load labels
    objects = readLabels(label_dir, i-1);
    
    for o=1:numel(objects)
        [corners, face_idx] = computeBox3D(objects(o), P);
        orientation = computeOrientation3D(objects(o), P);
        alpha = rad2deg(objects(o).alpha);
        
        if isempty(corners)
            objects(o).type = 'DontCare';
            objects(o).truncation = -1;
            objects(o).occlusion = -1;
            objects(o).alpha = -10;
            anchor_class = -1;
            x_l = -10000;
            y_l = -10000;
            x_m = -10000;
            y_m = -10000;
            x_r = -10000;
            y_r = -10000;
            x_t = -10000;
            y_t = -10000;
            
            x1 = objects(o).x1;
            y1 = objects(o).y1;
            x2 = objects(o).x2;
            y2 = objects(o).y2;
        else
            if alpha >= 0 && alpha < 90
                anchor_class = 0;
                ids = [3, 2, 1];
                x_l = corners(1, 3);
                y_l = corners(2, 3);
                x_m = corners(1, 2);
                y_m = corners(2, 2);
                x_r = corners(1, 1);
                y_r = corners(2, 1);
                x_t = corners(1, 6);
                y_t = corners(2, 6);
            elseif alpha >= 90 && alpha < 180
                anchor_class = 1;
                ids = [2, 1, 4];
                x_l = corners(1, 2);
                y_l = corners(2, 2);
                x_m = corners(1, 1);
                y_m = corners(2, 1);
                x_r = corners(1, 4);
                y_r = corners(2, 4);
                x_t = corners(1, 5);
                y_t = corners(2, 5);
            elseif alpha >= -90 && alpha < 0
                anchor_class = 2;
                ids = [4, 3, 2];
                x_l = corners(1, 4);
                y_l = corners(2, 4);
                x_m = corners(1, 3);
                y_m = corners(2, 3);
                x_r = corners(1, 2);
                y_r = corners(2, 2);
                x_t = corners(1, 7);
                y_t = corners(2, 7);
            elseif alpha >= -180 && alpha < -90
                anchor_class = 3;
                ids = [1, 4, 3];
                x_l = corners(1, 1);
                y_l = corners(2, 1);
                x_m = corners(1, 4);
                y_m = corners(2, 4);
                x_r = corners(1, 3);
                y_r = corners(2, 3);
                x_t = corners(1, 8);
                y_t = corners(2, 8);
            end
            x1 = min(corners(1, :));
            y1 = min(corners(2, :));
            x2 = max(corners(1, :));
            y2 = max(corners(2, :));
        end
        
        fprintf(f, '%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n', objects(o).type, ...
            objects(o).truncation, objects(o).occlusion, objects(o).alpha, ...
            x1, y1, x2, y2, x_l, y_l, x_m, y_m, x_r, y_r, x_t, y_t, ...
            objects(o).h, objects(o).w, objects(o).l, anchor_class);
        fprintf('%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n', objects(o).type, ...
            objects(o).truncation, objects(o).occlusion, objects(o).alpha, ...
            x1, y1, x2, y2, x_l, y_l, x_m, y_m, x_r, y_r, x_t, y_t, ...
            objects(o).h, objects(o).w, objects(o).l, anchor_class);
    end
    fclose(f);
    fprintf('Done with annotation file %d/%d...\n', i, length(files));
end
