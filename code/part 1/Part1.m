close all; clear all; clc;
import myfunctions.*

%% definitions

A = [0 0; 8 0; 8 1; 0 1];

B01 = [0 29; 32 29; 32 32; 0 32];
B02 = [0 0; 1 0; 1 32; 0 32];
B03 = [0 0; 32 0; 32 1; 0 1];
B04 = [31 0; 32 0; 32 32; 31 32];

B1 = [0 18; 10 18; 10 19; 0 19];
B2 = [17 17; 18 17; 18 29; 17 29];
B3 = [25 18; 32 18; 32 19; 25 19];
B4 = [0 14; 19 14; 19 15; 0 15];
B5 = [24 13; 32 13; 32 15; 24 15];
B6 = [10 19; 12 19; 12 20; 10 20];
B7 = [23 19; 25 19; 25 20; 23 20];

B_names = {'B01', 'B02', 'B03', 'B04', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'};
B_list = {B01, B02, B03, B04, B1, B2, B3, B4, B5, B6, B7};

% Plot original obstacle map
figure; axis equal; grid minor; hold on;
xlabel('X'); ylabel('Y');
title('Map of Obstacles');

for idx = 1:length(B_list)
    B = B_list{idx};
    fill(B(:,1), B(:,2), 'c', 'FaceAlpha', 0.3, 'EdgeColor', 'b');
    text(mean(B(:,1)), mean(B(:,2)), B_names{idx}, 'HorizontalAlignment', 'center', 'FontSize', 12);
end

hold off;

%% Q1

obstacle_list = {B01};
obstacle_name = {'B01'};

slices = cspace_slices_multiple(A, obstacle_list, 128);

% Define list of layers to plot
layers_to_plot = [1, 8, 16, 32]; % Change as needed

for layer = layers_to_plot
    figure; axis equal; grid minor; hold on;
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

slices = cspace_slices_multiple(A, obstacle_list, 128);

% Define list of layers to plot
layers_to_plot = [1, 8, 16, 32]; % Change as needed

for layer = layers_to_plot
    figure; axis equal; grid minor; hold on;
    title(sprintf('C-obstacle slice θ Layer %d', layer));

    for idx = 1:length(obstacle_list)
        obstacle = obstacle_list{idx};
        slice_points = slices{layer, idx};
        fill(slice_points(:,1), slice_points(:,2), 'r', 'FaceAlpha', 0.3);
        text(mean(obstacle(:,1)), mean(obstacle(:,2)), obstacle_name{idx}, 'HorizontalAlignment', 'center', 'FontSize', 12);
    end

    hold off;
end

%% Q3 – Discretized C-space Grid with COLORED BOUNDARIES and BLACK-WHITE MAP

% Parameters
grid_size = 65;
theta_layers = 128;
x_min = 0; x_max = 32;
y_min = 0; y_max = 32;

% Obstacles
obstacle_list = {B01, B02, B03, B04, B1, B2, B3, B4, B5, B6, B7};

% Generate C-space slices
slices = cspace_slices_multiple(A, obstacle_list, 128);

% Initialize grid (1: obstacle index, 0: free)
cspace_grid = zeros(grid_size, grid_size, theta_layers);
cspace_bw = false(grid_size, grid_size, theta_layers);  % for black-white

% Process each θ layer
for k = 1:theta_layers
    fprintf('Processing θ layer %d/%d\n', k, theta_layers);

    % For unioned black-white version
    union_poly = polyshape();

    % Process each obstacle
    for j = 1:length(obstacle_list)
        verts = slices{k, j};
        if size(verts, 1) > 2
            % Draw colored boundary version
            cspace_grid(:,:,k) = draw_polygon_boundary_lines_colored(cspace_grid(:,:,k), verts, x_min, x_max, y_min, y_max, grid_size, j);

            % Add to union for BW version
            try
                p = polyshape(verts);
                union_poly = union(union_poly, p);
            catch
                continue
            end
        end
    end

    % Fill BW grid from merged polygon
    cspace_bw(:,:,k) = fill_polygon_from_union(union_poly, x_min, x_max, y_min, y_max, grid_size)';
end

% Define colors for obstacles
colors = [
    0, 0, 0;        % Black (unused)
    1, 0, 0;        % Red for B01
    0, 1, 0;        % Green for B02
    0, 0, 1;        % Blue for B03
    1, 1, 0;        % Yellow for B04
    1, 0, 1;        % Magenta for B1
    0, 1, 1;        % Cyan for B2
    0.5, 0.5, 0.5;  % Gray for B3
    1, 0.5, 0;      % Orange for B4
    0.5, 0, 0.5;    % Purple for B5
    0, 0.5, 0.5;    % Teal for B6
    0.5, 0.5, 0;    % Olive for B7
];

% Plot layers
layers_to_plot = [1, 8, 16, 32];

for layer = layers_to_plot
    % Print matrix
    fprintf('\nMatrix: C-space occupancy of layer #%d\n', layer);
    for i = size(cspace_bw,1):-1:1
        for j = 1:size(cspace_bw,2)
            fprintf('%d', cspace_bw(i,j,layer));
        end
        fprintf('\n');
    end

    % Black-White version (like the example)
    x_res = (x_max - x_min) / (grid_size - 1);
    y_res = (y_max - y_min) / (grid_size - 1);
    [X, Y] = meshgrid(x_min:x_res:x_max, y_min:y_res:y_max);

    figure;
    pcolor(X, Y, double(cspace_bw(:,:,layer)));
    shading flat; colormap([1 1 1; 0 0 0]);
    axis equal tight;
    title(sprintf('Discretized C-obstacle Grid - $\\theta$ Layer %d (BW)', layer), ...
    'Interpreter', 'latex', 'FontWeight', 'bold');    xlabel('X'); ylabel('Y');
    set(gca, 'YDir', 'normal');
    colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Free', 'Obstacle'});

    % Colored boundary plot
    figure; hold on; axis equal tight;
    title(sprintf('$CB_{\\text{\\theta}}$ - Layer %d', layer), ...
    'Interpreter', 'latex', 'FontWeight', 'bold');
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    grid on;

    % Plot each obstacle's boundary
    grid_matrix = cspace_grid(:,:,layer);
    legend_handles = []; legend_labels = {};

    for obstacle_idx = 1:length(obstacle_list)
        [i_idx, j_idx] = find(grid_matrix == obstacle_idx);
        if ~isempty(i_idx)
            x_coords = x_min + (i_idx - 1) * x_res;
            y_coords = y_min + (j_idx - 1) * y_res;
            h = scatter(x_coords, y_coords, 50, colors(obstacle_idx+1,:), 'filled');
            legend_handles(end+1) = h;
            legend_labels{end+1} = B_names{obstacle_idx};
        end
    end

    if ~isempty(legend_handles)
        legend(legend_handles, legend_labels, 'Location', 'eastoutside');
    end

    xlim([x_min x_max]); ylim([y_min y_max]);
    xticks(0:2:32); yticks(0:2:32);
end

% Save result
save('cspace_boundary_grid_combined.mat', 'cspace_grid', 'cspace_bw', 'grid_size', 'x_min', 'x_max', 'y_min', 'y_max');

%%
function grid = draw_polygon_boundary_lines_colored(grid, vertices, x_min, x_max, y_min, y_max, N, obstacle_id)
    % Draw lines between consecutive vertices to form polygon boundary
    % Now assigns obstacle_id instead of just 1
    x_res = (x_max - x_min) / (N - 1);
    y_res = (y_max - y_min) / (N - 1);
    
    % Draw lines between consecutive vertices
    for i = 1:size(vertices, 1)
        % Get current vertex and next vertex (wrap around to close polygon)
        x1 = vertices(i, 1);
        y1 = vertices(i, 2);
        if i == size(vertices, 1)
            x2 = vertices(1, 1);  % Close the polygon
            y2 = vertices(1, 2);
        else
            x2 = vertices(i + 1, 1);
            y2 = vertices(i + 1, 2);
        end
        
        % Draw line from (x1,y1) to (x2,y2) using Bresenham's algorithm
        grid = draw_line_colored(grid, x1, y1, x2, y2, x_min, x_max, y_min, y_max, N, obstacle_id);
    end
end

%%
function grid = draw_line_colored(grid, x1, y1, x2, y2, x_min, x_max, y_min, y_max, N, obstacle_id)
    % Draw a line using Bresenham's algorithm
    % Now assigns obstacle_id instead of just 1
    x_res = (x_max - x_min) / (N - 1);
    y_res = (y_max - y_min) / (N - 1);
    
    % Convert to grid coordinates
    grid_x1 = round((x1 - x_min) / x_res) + 1;
    grid_y1 = round((y1 - y_min) / y_res) + 1;
    grid_x2 = round((x2 - x_min) / x_res) + 1;
    grid_y2 = round((y2 - y_min) / y_res) + 1;
    
    % Bresenham's line algorithm
    dx = abs(grid_x2 - grid_x1);
    dy = abs(grid_y2 - grid_y1);
    
    if grid_x1 < grid_x2
        sx = 1;
    else
        sx = -1;
    end
    
    if grid_y1 < grid_y2
        sy = 1;
    else
        sy = -1;
    end
    
    err = dx - dy;
    x = grid_x1;
    y = grid_y1;
    
    while true
        % Mark the point if it's within bounds
        if x >= 1 && x <= N && y >= 1 && y <= N
            grid(x, y) = obstacle_id;  % Mark with obstacle ID
        end
        
        % Check if we've reached the end point
        if x == grid_x2 && y == grid_y2
            break;
        end
        
        e2 = 2 * err;
        if e2 > -dy
            err = err - dy;
            x = x + sx;
        end
        if e2 < dx
            err = err + dx;
            y = y + sy;
        end
    end
end
%%
function grid = fill_polygon_from_union(poly, x_min, x_max, y_min, y_max, N)
    % Fill a binary grid (N x N) with 1 inside the polygon and 0 outside

    % Create empty binary grid
    grid = false(N, N);

    % Resolution per cell
    x_res = (x_max - x_min) / (N - 1);
    y_res = (y_max - y_min) / (N - 1);

    % Grid centers
    for i = 1:N
        for j = 1:N
            x = x_min + (i - 1) * x_res;
            y = y_min + (j - 1) * y_res;
            if isinterior(poly, x, y)
                grid(i, j) = true;
            end
        end
    end
end