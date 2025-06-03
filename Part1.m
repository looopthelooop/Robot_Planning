close all; clear all; clc;
import myfunctions.*

%% definitions

A = [0 0; 8 0; 8 1; 0 1];

B01 = [0 30; 31 30; 31 31; 0 31];
B02 = [0 1; 1 1; 1 30; 0 30];
B03 = [0 0; 31 0; 31 1; 0 1];
B04 = [30 1; 31 1; 31 30; 30 30];

B1 = [0 18; 10 18; 10 19; 0 19];
B2 = [17 17; 18 17; 18 30; 17 30];
B3 = [24 18; 30 18; 30 19; 24 19];
B4 = [0 14; 19 14; 19 15; 0 15];
B5 = [23 13; 31 13; 31 15; 23 15];
B6 = [10 19; 12 19; 12 20; 10 20];
B7 = [22 19; 24 19; 24 20; 22 20];

B_names = {'B01', 'B02', 'B03', 'B04', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'};
B_list = {B01, B02, B03, B04, B1, B2, B3, B4, B5, B6, B7};

% Plot
figure; axis equal; grid on; hold on;
xlabel('X'); ylabel('Y');
title('Map of Obstacles');

for idx = 1:length(B_list)
    B = B_list{idx};
    fill(B(:,1), B(:,2), 'c', 'FaceAlpha', 0.3, 'EdgeColor', 'b');
    % Use the corresponding name from B_names
    text(mean(B(:,1)), mean(B(:,2)), B_names{idx}, 'HorizontalAlignment', 'center', 'FontSize', 12);
end

hold off;

%% Q1

obstacle_list = {B01};
obstacle_name = {'B01'};

slices = cspace_slices_multiple(A, obstacle_list);

% Define list of layers to plot
layers_to_plot = [1, 8, 16, 32]; % Change as needed

for layer = layers_to_plot
    figure; axis equal; grid on; hold on;
    title(sprintf('C-obstacle slice θ Layer %d', layer));

    for idx = 1:length(obstacle_list)
        obstacle = obstacle_list{idx};
        slice_points = slices{layer, idx};
        fill(slice_points(:,1), slice_points(:,2), 'r', 'FaceAlpha', 0.3);
        text(mean(obstacle(:,1)), mean(obstacle(:,2)), obstacle_name{idx}, 'HorizontalAlignment', 'center', 'FontSize', 12);
    end

    hold off;
end


%% Q2

obstacle_list = {B01, B02, B03, B04, B1, B2, B3, B4, B5, B6, B7};
obstacle_name = {'B01', 'B02', 'B03', 'B04', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'};

slices = cspace_slices_multiple(A, obstacle_list);

% Define list of layers to plot
layers_to_plot = [1, 8, 16, 32]; % Change as needed

for layer = layers_to_plot
    figure; axis equal; grid on; hold on;
    title(sprintf('C-obstacle slice θ Layer %d', layer));

    for idx = 1:length(obstacle_list)
        obstacle = obstacle_list{idx};
        slice_points = slices{layer, idx};
        fill(slice_points(:,1), slice_points(:,2), 'r', 'FaceAlpha', 0.3);
        text(mean(obstacle(:,1)), mean(obstacle(:,2)), obstacle_name{idx}, 'HorizontalAlignment', 'center', 'FontSize', 12);
    end

    hold off;
end

%% Q3 - Discretized C-space Boundary Grid (32x32x32) with Polygon Rasterization

grid_size = 32;
x_range = linspace(-5, 35, grid_size);
y_range = linspace(-5, 35, grid_size);
theta_values = linspace(0, 2*pi - 2*pi/grid_size, grid_size);

cspace_grid = zeros(grid_size, grid_size, grid_size);

for k = 1:grid_size
    % Initialize blank mask for this layer
    layer_mask = false(grid_size, grid_size);
    
    for obs_idx = 1:length(obstacle_list)
        slice = slices{k, obs_idx};
        
        % Clean slice: remove duplicate vertices, ensure closure
        slice = unique(slice, 'rows', 'stable');
        if ~isequal(slice(1,:), slice(end,:))
            slice = [slice; slice(1,:)];
        end
        
        % Scale polygon coordinates to grid indices
        x_scaled = round( (slice(:,1) - min(x_range)) / (max(x_range)-min(x_range)) * (grid_size-1) ) + 1;
        y_scaled = round( (slice(:,2) - min(y_range)) / (max(y_range)-min(y_range)) * (grid_size-1) ) + 1;
        
        % Clip indices to grid bounds
        x_scaled = min(max(x_scaled,1), grid_size);
        y_scaled = min(max(y_scaled,1), grid_size);
        
        % Create mask for the polygon boundary (poly2mask)
        mask = poly2mask(x_scaled, y_scaled, grid_size, grid_size);
        
        % Combine mask into layer
        layer_mask = layer_mask | mask;
    end
    
    % Assign combined layer mask to cspace_grid
    cspace_grid(:,:,k) = layer_mask;
end

% Plot selected layers
layers_to_plot = [1, 8, 16, 32];
[X, Y] = meshgrid(x_range, y_range);
for layer = layers_to_plot
    figure; axis equal; grid on; hold on;
    title(sprintf('Discretized C-space Boundary Slice θ Layer %d', layer));
    imagesc(x_range, y_range, cspace_grid(:,:,layer));
    colormap([1 1 1; 0 0.7 0.7]);
    caxis([0 1]);
    set(gca,'YDir','normal');
    hold off;
end
