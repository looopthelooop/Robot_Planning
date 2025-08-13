import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import heapq
import math
import time
from matplotlib.patches import Rectangle, Polygon


def load_cspace(path):
    # Load and transpose C-space from MATLAB file

    cspace = loadmat(path)['cspace_bw']
    # Transpose from (y,x,theta) to (x,y,theta)
    return np.transpose(cspace, (1, 0, 2))


def heuristic(a, b):
    # Euclidean distance heuristic

    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def move_cost(dx, dy, dtheta):
    # Cost function for movement - position + small cost for rotation

    pos_cost = math.sqrt(dx*dx + dy*dy) if (dx or dy) else 0
    rot_cost = 0.1 * abs(dtheta)
    return pos_cost + rot_cost


def get_neighbors(node):
    # Get neighboring states
    x, y, theta = node
    
    moves = [
        # Basic moves
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),  # Cardinal
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),  # Diagonal
        (0, 0, 1), (0, 0, -1),  # Rotate only
        # Combined moves
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ]
    
    neighbors = []
    for dx, dy, dtheta in moves:
        nx, ny, nt = x + dx, y + dy, (theta + dtheta) % 128
        neighbors.append((nx, ny, nt, dx, dy, dtheta))
    
    return neighbors


def a_star(cspace, start, goal):
    """A* pathfinding algorithm"""
    open_set = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    came_from = {}
    visited = set()

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)

        # Goal check - allow orientation tolerance
        if (current[0] == goal[0] and current[1] == goal[1] and current[2] == goal[2]):

            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, visited

        # Explore neighbors
        for nx, ny, nt, dx, dy, dtheta in get_neighbors(current):
            # Check bounds
            if not (0 <= nx < cspace.shape[0] and 0 <= ny < cspace.shape[1]):
                continue
            
            # Check collision
            if cspace[nx, ny, nt] == 1:
                continue
                
            neighbor = (nx, ny, nt)
            new_cost = g_score[current] + move_cost(dx, dy, dtheta)
            
            if neighbor not in g_score or new_cost < g_score[neighbor]:
                g_score[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return None, visited


def grid_to_world(grid_coord):
    """Convert grid index to world coordinate"""
    return grid_coord * 32.0 / 64.0


def world_to_grid(world_coord):
    """Convert world coordinate to grid index"""
    return int(round(world_coord * 64.0 / 32.0))


def draw_robot(ax, x_grid, y_grid, theta, color='blue', alpha=1.0):
    """Draw robot rectangle with orientation arrow"""
    # Convert to world coordinates
    world_x = grid_to_world(x_grid)
    world_y = grid_to_world(y_grid)
    
    # Robot size and angle
    robot_width, robot_height = 8.0, 1.0
    angle_deg = theta * 360.0 / 128.0
    
    # Draw robot body
    rect = Rectangle((world_x, world_y), robot_width, robot_height, 
                    angle=angle_deg, color=color, alpha=alpha)
    ax.add_patch(rect)
    
    # Draw orientation arrow
    angle_rad = np.radians(angle_deg)
    arrow_length = robot_width * 0.6
    end_x = world_x + arrow_length * np.cos(angle_rad)
    end_y = world_y + arrow_length * np.sin(angle_rad)
    ax.plot([world_x, end_x], [world_y, end_y], 
            color='white', linewidth=2, alpha=alpha)


def create_apartment_obstacles():
    """Create the apartment layout - COORDINATES VERIFIED"""
    obstacles = [
        # Boundary walls
        [[0, 29], [32, 29], [32, 32], [0, 32]],  # Top wall
        [[0, 0], [1, 0], [1, 32], [0, 32]],      # Left wall
        [[0, 0], [32, 0], [32, 1], [0, 1]],      # Bottom wall
        [[31, 1], [32, 1], [32, 32], [31, 32]],  # Right wall
        
        # Interior obstacles
        [[0, 18], [10, 18], [10, 19], [0, 19]],   # B1
        [[17, 17], [18, 17], [18, 29], [17, 29]], # B2
        [[25, 18], [32, 18], [32, 19], [25, 19]], # B3
        [[0, 14], [19, 14], [19, 15], [0, 15]],   # B4
        [[24, 13], [32, 13], [32, 15], [24, 15]], # B5
        [[10, 19], [12, 19], [12, 20], [10, 20]], # B6
        [[23, 19], [25, 19], [25, 20], [23, 20]], # B7
    ]
    
    return obstacles


def visualize_path(cspace, path, visited):
    """Display the robot path with obstacles and statistics"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw apartment obstacles
    for obstacle in create_apartment_obstacles():
        polygon = Polygon(obstacle, facecolor='gray', edgecolor='black', alpha=0.8)
        ax.add_patch(polygon)
    
    # Draw path if found
    if path:
        # Draw robot at each position
        for i, (x_grid, y_grid, theta) in enumerate(path):
            if i == 0:
                color, alpha = 'green', 1.0    # Start
            elif i == len(path) - 1:
                color, alpha = 'red', 1.0      # Goal
            else:
                # Color gradient along path
                ratio = i / (len(path) - 1)
                color = plt.cm.plasma(ratio)
                alpha = 0.4
            
            draw_robot(ax, x_grid, y_grid, theta, color=color, alpha=alpha)
        
        # Draw path line
        path_x = [grid_to_world(x) for x, y, theta in path]
        path_y = [grid_to_world(y) for x, y, theta in path]
        ax.plot(path_x, path_y, 'k-', linewidth=2, alpha=0.8, label='Path')
        
        # Add start/goal labels
        ax.text(path_x[0] + 1, path_y[0] + 0.5, 'START', 
                color='white', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.9))
        ax.text(path_x[-1] + 1, path_y[-1] + 0.5, 'GOAL', 
                color='white', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.9))
        
        # Show statistics
        total_distance = sum(math.sqrt((path_x[i+1]-path_x[i])**2 + (path_y[i+1]-path_y[i])**2) 
                           for i in range(len(path_x)-1))
        stats = f'Steps: {len(path)}\nDistance: {total_distance:.1f}\nVisited: {len(visited)}'
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Set up plot
    ax.set_title('Robot Path Planning', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if path:
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function - load data, plan path, visualize results"""
    
    # Load C-space
    cspace = load_cspace('C:\Repos\Robot_Planning\cspace_boundary_grid_combined.mat')
    print(f"Loaded C-space: {cspace.shape}")

    # Define start and goal positions
    start_world = (4, 24, 0)   # World coordinates
    goal_world = (4, 8, 0)
    
    # Convert to grid coordinates
    start = (world_to_grid(start_world[0]), world_to_grid(start_world[1]), start_world[2])
    goal = (world_to_grid(goal_world[0]), world_to_grid(goal_world[1]), goal_world[2])
    
    print(f"Start: {start_world} → {start}")
    print(f"Goal: {goal_world} → {goal}")
    
    # Validate positions
    if (not (0 <= start[0] < cspace.shape[0] and 0 <= start[1] < cspace.shape[1]) or
        not (0 <= goal[0] < cspace.shape[0] and 0 <= goal[1] < cspace.shape[1])):
        print("Error: Start or goal out of bounds!")
        return
    
    if cspace[start[0], start[1], start[2]] == 1 or cspace[goal[0], goal[1], goal[2]] == 1:
        print("Error: Start or goal position blocked!")
        return
    
    # Find path
    print("Running A* pathfinding...")
    path, visited = a_star(cspace, start, goal)
    
    # Show results
    if path:
        print(f"SUCCESS! Found path with {len(path)} steps")
        visualize_path(cspace, path, visited)
    else:
        print("FAILED! No path found")
        visualize_path(cspace, None, visited)


if __name__ == '__main__':
    main()