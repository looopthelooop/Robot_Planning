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

%% Q3 – Discretized C-space Grid using unioned polyshape

% Parameters
grid_size = 32;
theta_layers = 32;
x_min = 0; x_max = 31;
y_min = 0; y_max = 31;

% Obstacles
obstacle_list = {B01, B02, B03, B04, B1, B2, B3, B4, B5, B6, B7};

% Generate C-space slices
slices = cspace_slices_multiple(A, obstacle_list);

% Initialize grid
cspace_grid = false(grid_size, grid_size, theta_layers);

% Process each θ layer
for k = 1:theta_layers
    fprintf('Processing θ layer %d/%d\n', k, theta_layers);
    union_poly = polyshape();  % Initialize empty union polygon

    for j = 1:length(obstacle_list)
        verts = slices{k, j};
        if size(verts, 1) > 2
            cleaned = clean_polygon(verts);
            try
                p = polyshape(cleaned);
                union_poly = union(union_poly, p);
            catch
                continue
            end
        end
    end

    % Fill grid from merged polygon
    cspace_grid(:,:,k) = fill_polygon_from_union(union_poly, x_min, x_max, y_min, y_max, grid_size)';
end

% Plot selected θ layers
layers_to_plot = [1, 8, 16, 32];
x_res = (x_max - x_min) / (grid_size - 1);
y_res = (y_max - y_min) / (grid_size - 1);
[X, Y] = meshgrid(x_min:x_res:x_max, y_min:y_res:y_max);

for layer = layers_to_plot
    figure;
    pcolor(X, Y, double(cspace_grid(:,:,layer)));
    shading flat; colormap([1 1 1; 0 0 0]);
    axis equal tight;
    title(sprintf('Discretized C-obstacle Grid - θ Layer %d', layer));
    xlabel('X'); ylabel('Y');
    set(gca, 'YDir', 'normal');
    colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Free', 'Obstacle'});
end

% Save the result
save('cspace_grid.mat', 'cspace_grid', 'grid_size', 'x_min', 'x_max', 'y_min', 'y_max');

%% Helper: Clean polygon vertices (remove duplicates)
function cleaned = clean_polygon(verts)
    diff_v = diff([verts; verts(1,:)], 1, 1);
    keep = any(abs(diff_v) > 1e-10, 2);
    cleaned = verts(keep, :);
end

%% Helper: Fill grid based on polygon intersection
function grid = fill_polygon_from_union(union_poly, x_min, x_max, y_min, y_max, N)
    x_res = (x_max - x_min) / (N - 1);
    y_res = (y_max - y_min) / (N - 1);
    grid = false(N, N);

    for i = 1:N
        for j = 1:N
            x0 = x_min + (i - 1) * x_res;
            x1 = x0 + x_res;
            y0 = y_min + (j - 1) * y_res;
            y1 = y0 + y_res;
            cell_poly = polyshape([x0 x1 x1 x0], [y0 y0 y1 y1]);

            if overlaps(union_poly, cell_poly)
                grid(i,j) = true;
            end
        end
    end
end
