function all_slices = cspace_slices_multiple(A, B_list, theta_layers)
    % A: Nx2 vertices of robot A
    % B_list: cell array of Mx2 obstacles
    theta_values = linspace(0, 2*pi - 2*pi/theta_layers, theta_layers);
    all_slices = cell(theta_layers, length(B_list)); % Initialize cell array to hold slices

    for k = 1:theta_layers
        theta = theta_values(k);
        R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
        rotated_A = (R * A')'; % Rotate
        
        for idx = 1:length(B_list)
            B = B_list{idx};
            B_col = B'; % 2xM
            
            % Minkowski difference (B - rotated A)
            diff_points = [];
            for i = 1:size(B_col,2)
                for j = 1:size(rotated_A,1)
                    diff_points = [diff_points, B_col(:,i) - rotated_A(j,:)'];
                end
            end
            points = diff_points';
            
            % Compute convex hull
            K = convhull(points(:,1), points(:,2));
            all_slices{k, idx} = points(K, :);
        end
    end
end