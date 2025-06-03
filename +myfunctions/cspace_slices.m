function cspace_slices(A, B)
    % A: Nx2 vertices of robot A
    % B: Mx2 vertices of obstacle B
    
    % Convert A to column format
    A_col = A'; % 2xN
    B_col = B'; % 2xM
    
    % Compute Minkowski difference (all combinations of B - A)
    diff_points = [];
    for i = 1:size(B_col,2)
        for j = 1:size(A_col,2)
            diff_points = [diff_points, B_col(:,i) - A_col(:,j)];
        end
    end
    
    % Convert to 2D point matrix
    points = diff_points';
    
    % Compute convex hull of points
    K = convhull(points(:,1), points(:,2));
   
end
