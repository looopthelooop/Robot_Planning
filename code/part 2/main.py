import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import heapq
import math
import time
from matplotlib.patches import Rectangle, Polygon


def load_cspace(path):
    """Load the C-space from MATLAB file"""
    cspace = loadmat(path)['cspace_bw']
    print(f"Original C-space shape: {cspace.shape}")
    
    # IMPORTANT: Check if we need to transpose to match coordinate systems
    # MATLAB often uses (row, col) which corresponds to (y, x) in Cartesian
    # We may need to transpose the first two dimensions
    
    # Let's assume the C-space is stored as (y, x, theta) and we need (x, y, theta)
    cspace_corrected = np.transpose(cspace, (1, 0, 2))
    print(f"Corrected C-space shape: {cspace_corrected.shape}")
    
    return cspace_corrected


def heuristic(a, b):
    """Euclidean distance heuristic (better than Manhattan for this case)"""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def move_cost(dx, dy, dtheta):
    """Cost function for movement"""
    # Euclidean distance for position + small cost for rotation
    pos_cost = math.sqrt(dx*dx + dy*dy) if (dx or dy) else 0
    rot_cost = 0.1 * abs(dtheta)
    return pos_cost + rot_cost


def get_neighbors(node):
    """Get all possible neighboring states with improved movement model"""
    x, y, theta = node
    
    # More realistic movement patterns
    moves = [
        # Cardinal directions
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        # Diagonal movements
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        # Rotation only
        (0, 0, 1), (0, 0, -1),
        # Combined movement with rotation (more realistic)
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ]
    
    neighbors = []
    for dx, dy, dtheta in moves:
        nx = x + dx
        ny = y + dy
        nt = (theta + dtheta) % 32
        neighbors.append((nx, ny, nt, dx, dy, dtheta))
    
    return neighbors


def a_star(cspace, start, goal):
    """A* pathfinding algorithm with improved implementation"""
    print(f"Searching from {start} to {goal}")
    start_time = time.time()
    
    open_set = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    came_from = {}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)

        # Goal check - allow some tolerance in orientation
        if (current[0] == goal[0] and current[1] == goal[1] and 
            abs(current[2] - goal[2]) <= 1):
            
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            
            elapsed = time.time() - start_time
            print(f"Path found! Length: {len(path[::-1])}, Time: {elapsed:.2f}s, Visited: {len(visited)}")
            return path[::-1], visited

        for nx, ny, nt, dx, dy, dtheta in get_neighbors(current):
            # Check bounds - CORRECTED coordinate system
            if not (0 <= nx < cspace.shape[0] and 0 <= ny < cspace.shape[1]):
                continue
            
            # Check collision - using corrected indexing
            if cspace[nx, ny, nt] == 1:
                continue
                
            neighbor = (nx, ny, nt)
            new_cost = g_score[current] + move_cost(dx, dy, dtheta)
            
            if neighbor not in g_score or new_cost < g_score[neighbor]:
                g_score[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    print(f"No path found after visiting {len(visited)} nodes")
    return None, visited


def grid_to_world(grid_coord, grid_size=64):
    """Convert grid index to world coordinate"""
    return grid_coord * 31.0 / (grid_size - 1)


def world_to_grid(world_coord, grid_size=64):
    """Convert world coordinate to grid index"""
    return int(round(world_coord * (grid_size - 1) / 31.0))


def draw_robot(ax, x_grid, y_grid, theta, color='blue', alpha=1.0, grid_size=64):
    """Draw robot with CORRECTED coordinate system
    
    Args:
        x_grid, y_grid: Grid indices (0-63) - NOW CORRECTLY MAPPED
        theta: Orientation (0-31)
    """
    # Convert grid indices to world coordinates
    world_x = grid_to_world(x_grid, grid_size)
    world_y = grid_to_world(y_grid, grid_size)
    
    # Robot dimensions in world coordinates
    robot_width = 8.0
    robot_height = 1.0
    
    # Calculate rotation angle
    angle_deg = (theta * 360.0 / 32.0)
    
    # Create rectangle at correct position
    rect = Rectangle((world_x, world_y), robot_width, robot_height, 
                    angle=angle_deg, 
                    color=color, alpha=alpha, fill=True)
    ax.add_patch(rect)
    
    # Add orientation indicator
    forward_length = robot_width * 0.6
    angle_rad = np.radians(angle_deg)
    forward_x = world_x + forward_length * np.cos(angle_rad)
    forward_y = world_y + forward_length * np.sin(angle_rad)
    ax.plot([world_x, forward_x], [world_y, forward_y], 
            color='white', linewidth=2, alpha=alpha)


def create_apartment_layout():
    """Create the apartment layout - COORDINATES VERIFIED"""
    obstacles = [
        # Boundary walls
        [[0, 30], [31, 30], [31, 31], [0, 31]],  # Top wall
        [[0, 1], [1, 1], [1, 30], [0, 30]],      # Left wall  
        [[0, 0], [31, 0], [31, 1], [0, 1]],      # Bottom wall
        [[30, 1], [31, 1], [31, 30], [30, 30]],  # Right wall
        
        # Interior obstacles
        [[0, 18], [10, 18], [10, 19], [0, 19]],   # B1
        [[17, 17], [18, 17], [18, 30], [17, 30]], # B2
        [[24, 18], [30, 18], [30, 19], [24, 19]], # B3
        [[0, 14], [19, 14], [19, 15], [0, 15]],   # B4
        [[23, 13], [31, 13], [31, 15], [23, 15]], # B5
        [[10, 19], [12, 19], [12, 20], [10, 20]], # B6
        [[22, 19], [24, 19], [24, 20], [22, 20]], # B7
    ]
    
    return obstacles


def visualize_path_corrected(cspace, path, visited):
    """Visualize the robot path with CORRECTED coordinate system"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create and show the apartment layout
    obstacles = create_apartment_layout()
    
    # Draw obstacles
    for obstacle in obstacles:
        obstacle = np.array(obstacle)
        polygon = Polygon(obstacle, facecolor='gray', edgecolor='black', alpha=0.8)
        ax.add_patch(polygon)
    
    if path:
        print(f"Drawing path with {len(path)} steps")
        
        # Convert path from grid coordinates to world coordinates
        world_path = []
        for x_grid, y_grid, theta in path:
            world_x = grid_to_world(x_grid)
            world_y = grid_to_world(y_grid)
            world_path.append((world_x, world_y, theta))
        
        # Draw all robot positions along the path
        for i, (x_grid, y_grid, theta) in enumerate(path):
            if i == 0:
                color = 'green'
                alpha = 1.0
            elif i == len(path) - 1:
                color = 'red' 
                alpha = 1.0
            else:
                # Color gradient along path
                ratio = i / (len(path) - 1)
                color = plt.cm.plasma(ratio)
                alpha = 0.3 + 0.4 * ratio
            
            draw_robot(ax, x_grid, y_grid, theta, color=color, alpha=alpha)
        
        # Draw path line
        path_x = [world_x for world_x, world_y, theta in world_path]
        path_y = [world_y for world_x, world_y, theta in world_path]
        
        ax.plot(path_x, path_y, 'k-', linewidth=3, alpha=0.8, label='Robot Path')
        
        # Add labels
        start_x, start_y = path_x[0], path_y[0]
        goal_x, goal_y = path_x[-1], path_y[-1]
        
        ax.text(start_x + 1, start_y + 0.5, 'START', color='white', fontweight='bold', 
                ha='left', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.9))
        ax.text(goal_x + 1, goal_y + 0.5, 'GOAL', color='white', fontweight='bold', 
                ha='left', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.9))
        
        # Calculate statistics
        path_length = len(path)
        total_distance = sum(math.sqrt((world_path[i+1][0]-world_path[i][0])**2 + 
                                     (world_path[i+1][1]-world_path[i][1])**2) 
                           for i in range(len(world_path)-1))
        
        stats_text = f'Path Length: {path_length} steps\nTotal Distance: {total_distance:.1f} world units\nNodes Visited: {len(visited)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Set up coordinate system
    ax.set_title(f'Robot Path Planning - CORRECTED Coordinates\nRobot: 8×1 world units, {len(path) if path else 0} steps', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X coordinate (world units)', fontsize=14)
    ax.set_ylabel('Y coordinate (world units)', fontsize=14)
    
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set ticks
    ax.set_xticks(np.arange(0, 32, 2))
    ax.set_yticks(np.arange(0, 32, 2))
    
    plt.tight_layout()
    plt.show()


def debug_coordinate_system(cspace, test_points):
    """Debug function to verify coordinate system"""
    print("="*60)
    print("COORDINATE SYSTEM DEBUG")
    print("="*60)
    print(f"C-space shape: {cspace.shape}")
    print(f"Expected: (x_max, y_max, theta_max) = (64, 64, 32)")
    print()
    
    for i, (name, point) in enumerate(test_points.items()):
        x, y, theta = point
        print(f"{name}: Grid({x}, {y}, {theta})")
        print(f"  World coordinates: ({grid_to_world(x):.1f}, {grid_to_world(y):.1f})")
        print(f"  Bounds check: x={0 <= x < cspace.shape[0]}, y={0 <= y < cspace.shape[1]}")
        
        if (0 <= x < cspace.shape[0] and 0 <= y < cspace.shape[1]):
            collision = cspace[x, y, theta]
            print(f"  Collision status: {collision} ({'BLOCKED' if collision == 1 else 'FREE'})")
        else:
            print(f"  OUT OF BOUNDS!")
        print()


def main():
    """Main function with coordinate system verification"""
    
    # Load the C-space with coordinate correction
    try:
        cspace = load_cspace('cspace_boundary_grid_combined.mat')
    except FileNotFoundError:
        print("Error: Could not load C-space file.")
        print("Creating a simple test C-space for demonstration...")
        # Create a simple test C-space
        cspace = np.zeros((64, 64, 32))
        # Add some simple obstacles for testing
        cspace[20:40, 20:25, :] = 1  # Vertical wall
        cspace[10:15, 30:50, :] = 1  # Horizontal wall
    
    # Define test points for debugging
    test_points = {
        "Start": (world_to_grid(4), world_to_grid(16), 0),
        "Goal": (world_to_grid(8), world_to_grid(4), 0),
        "Center": (32, 32, 0),
        "Corner": (5, 5, 0)
    }
    
    # Debug coordinate system
    debug_coordinate_system(cspace, test_points)
    
    # Use the corrected start and goal
    start_world = (4, 24, 0)
    goal_world = (4, 8, 0)
    
    start = (world_to_grid(start_world[0]), world_to_grid(start_world[1]), start_world[2])
    goal = (world_to_grid(goal_world[0]), world_to_grid(goal_world[1]), goal_world[2])
    
    print(f"CORRECTED PATH PLANNING")
    print(f"Start: World{start_world} -> Grid{start}")
    print(f"Goal: World{goal_world} -> Grid{goal}")
    
    # Verify positions are valid
    if not (0 <= start[0] < cspace.shape[0] and 0 <= start[1] < cspace.shape[1]):
        print(f"ERROR: Start position out of bounds!")
        return
    if not (0 <= goal[0] < cspace.shape[0] and 0 <= goal[1] < cspace.shape[1]):
        print(f"ERROR: Goal position out of bounds!")
        return
    
    if cspace[start[0], start[1], start[2]] == 1:
        print("ERROR: Start position is in collision!")
        return
    if cspace[goal[0], goal[1], goal[2]] == 1:
        print("ERROR: Goal position is in collision!")
        return
    
    print("Positions verified as collision-free!")
    
    # Find path using corrected A*
    print("\nRunning CORRECTED A* pathfinding...")
    path, visited = a_star(cspace, start, goal)
    
    # Visualize with corrected coordinate system
    visualize_path_corrected(cspace, path, visited)
    
    if path:
        print(f"\nSUCCESS! Path found with {len(path)} steps")
        print("First few steps in world coordinates:")
        for i, (x_grid, y_grid, theta) in enumerate(path[:5]):
            world_x = grid_to_world(x_grid)
            world_y = grid_to_world(y_grid)
            angle_deg = theta * 360 / 32
            print(f"  Step {i}: World({world_x:5.1f}, {world_y:5.1f}, {angle_deg:6.1f}°)")
    else:
        print("\nPath planning failed - check coordinate system and obstacles")


if __name__ == '__main__':
    main()