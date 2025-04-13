import os
import random
from PIL import Image, ImageDraw

# Configuration
IMAGE_SIZE = 1000
GRID_ROWS = 6
GRID_COLS = 5
CELL_WIDTH = IMAGE_SIZE // GRID_COLS
CELL_HEIGHT = IMAGE_SIZE // GRID_ROWS
VERSION = "v3.5"
DATASET_DIR = "manga_grid_dataset_randomized" + "_" + VERSION
os.makedirs(DATASET_DIR, exist_ok=True)

def randomizer(x1,y1,x2,y2):
    width = x2 - x1
    height = y2 - y1
    probability = random.random()

    if probability < 0.1:
        x1 -= random.randint(0, width//5)
        x1 += random.randint(0, width//5)
        x2 -= random.randint(0, width//5)
        x2 += random.randint(0, width//5)

    if probability < 0.5:
        x1 -= random.randint(0, width//3)
        x1 += random.randint(0, width//3)
        x2 -= random.randint(0, width//3)
        x2 += random.randint(0, width//3)
    
    elif probability < 0.8:
        y1 -= random.randint(0, height//3)
        y1 += random.randint(0, height//3)
        y2 -= random.randint(0, height//3)
        y2 += random.randint(0, height//3)

    elif probability < 0.9:
        x1 -= random.randint(0, width//4)
        x1 += random.randint(0, width//4)
        y1 -= random.randint(0, height//4)
        y1 += random.randint(0, height//4)
        x2 -= random.randint(0, width//4)
        x2 += random.randint(0, width//4)
        y2 -= random.randint(0, height//4)
        y2 += random.randint(0, height//4)

    return x1 if x1 > 0 else 0+5, y1 if y1 > 0 else 0+5, x2 if x2 <= IMAGE_SIZE else IMAGE_SIZE - 5, y2 if y2 <= IMAGE_SIZE else IMAGE_SIZE - 5

        
def generate_valid_panels(rows, cols):
    """Generate panels with strict vertical span alignment rules"""
    GRID_ROWS = rows
    GRID_COLS = cols
    CELL_WIDTH = IMAGE_SIZE // GRID_COLS
    CELL_HEIGHT = IMAGE_SIZE // GRID_ROWS
    
    while True:
        occupied = set()
        panels = []
        prev_col_span = [1] * GRID_ROWS
        vertical_spans = {}  # Track ongoing vertical spans {col: (start_row, span_length)}

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if (row, col) in occupied:
                    continue

                row_span = 1
                col_span = 1
                can_vertical = False

                # Vertical span rules for last column
                if col == GRID_COLS-1:
                    # Check if we can continue existing vertical span
                    if col in vertical_spans:
                        vs_row, vs_span = vertical_spans[col]
                        if vs_row + vs_span == row:
                            # Continue existing vertical span with 80% probability
                            if random.random() < 0.8:
                                row_span = min(vs_span + 1, GRID_ROWS - vs_row)
                                can_vertical = True
                    else:
                        # Check if previous rows in this column are empty
                        valid_start = True
                        for r in range(row):
                            if (r, col) not in occupied:
                                valid_start = False
                                break
                        if valid_start and random.random() < 0.3:
                            row_span = min(2, GRID_ROWS - row)
                            can_vertical = True

                # Horizontal span rules (existing logic)
                if col < GRID_COLS-1 and (col == 0 or prev_col_span[row] > 1):
                    if random.random() < 0.2:
                        col_span = min(2, GRID_COLS - col)

                # Create coordinates
                x1 = col * CELL_WIDTH
                y1 = row * CELL_HEIGHT
                x2 = x1 + col_span * CELL_WIDTH
                y2 = y1 + row_span * CELL_HEIGHT

                # Validate placement
                valid = True
                for r in range(row, row + row_span):
                    for c in range(col, col + col_span):
                        if (r, c) in occupied:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Update vertical span tracking
                    if can_vertical and col == GRID_COLS-1:
                        if col in vertical_spans:
                            # Extend existing span
                            old_row, old_span = vertical_spans[col]
                            vertical_spans[col] = (old_row, old_span + row_span)
                        else:
                            # Start new span
                            vertical_spans[col] = (row, row_span)

                    # Coordinate randomization (existing logic)
                    prob = random.random()
                    if prob < 0.4:
                        final_coords = randomizer(x1, y1, x2, y2)
                    else:
                        reduce = 10 * prob
                        final_coords = (x1, y1, x2 - reduce, y2 - reduce)

                    panels.append({
                        "coords": final_coords,
                        "row": row,
                        "col": col,
                        "row_span": row_span,
                        "col_span": col_span
                    })

                    # Mark occupied cells
                    for r in range(row, row + row_span):
                        for c in range(col, col + col_span):
                            occupied.add((r, c))
                    prev_col_span[row] = col_span

        if len(occupied) == GRID_ROWS * GRID_COLS:
            return panels
        
def calculate_reading_order(panels):
    """Left-to-right topological sort with vertical span awareness"""
    # Create adjacency list and in-degree tracking
    graph = {id(p): [] for p in panels}
    in_degree = {id(p): 0 for p in panels}
    
    # Create position map and coverage tracker
    position_map = {}
    coverage_map = {}
    for p in panels:
        position_map[(p['row'], p['col'])] = p
        for r in range(p['row'], p['row'] + p['row_span']):
            for c in range(p['col'], p['col'] + p['col_span']):
                coverage_map[(r, c)] = p

    # Build dependency graph
    for panel in panels:
        # Vertical spans must come after all panels in their covered rows
        if panel['row_span'] > 1:
            for r in range(panel['row'], panel['row'] + panel['row_span']):
                for c in range(panel['col']):
                    if (r, c) in coverage_map:
                        predecessor = coverage_map[(r, c)]
                        if id(predecessor) not in graph:
                            graph[id(predecessor)] = []
                        graph[id(predecessor)].append(id(panel))
                        in_degree[id(panel)] += 1

        # Horizontal spans must come after their left neighbors
        if panel['col'] > 0:
            left_col = panel['col'] - 1
            if (panel['row'], left_col) in coverage_map:
                predecessor = coverage_map[(panel['row'], left_col)]
                graph[id(predecessor)].append(id(panel))
                in_degree[id(panel)] += 1

    # Kahn's algorithm with left-to-right priority
    queue = [p for p in panels if in_degree[id(p)] == 0]
    reading_order = []
    
    while queue:
        # Sort by row ascending, column ascending (left-to-right, top-to-bottom)
        queue.sort(key=lambda x: (x['row'], x['col']))
        current = queue.pop(0)
        reading_order.append(current)
        
        for neighbor_id in graph[id(current)]:
            neighbor = next(p for p in panels if id(p) == neighbor_id)
            in_degree[neighbor_id] -= 1
            if in_degree[neighbor_id] == 0:
                queue.append(neighbor)
    
    if len(reading_order) != len(panels):
        raise RuntimeError("Cyclic dependencies detected in layout")
    
    return [p['coords'] for p in reading_order], list(range(len(reading_order)))

def save_sample(sample_id, coords_list, sequence):
    """Save sample image and annotation"""
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), 'white')
    draw = ImageDraw.Draw(img)
    
    for idx, (x1, y1, x2, y2) in enumerate(coords_list):
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1+10, y1+10), str(sequence[idx]), fill='blue')
    
    img.save(os.path.join(DATASET_DIR, f"sample_{VERSION}_{sample_id}.png"))
    
    with open(os.path.join(DATASET_DIR, f"sample_{VERSION}_{sample_id}.txt"), "w") as f:
        for (x1, y1, x2, y2), order in zip(coords_list, sequence):
            x_center = ((x1 + x2) / 2) / IMAGE_SIZE
            y_center = ((y1 + y2) / 2) / IMAGE_SIZE
            width = (x2 - x1) / IMAGE_SIZE
            height = (y2 - y1) / IMAGE_SIZE
            f.write(f"{order} {x_center} {y_center} {width} {height}\n")

def generate_dataset(num_samples):
    """Generate dataset with variations"""
    for sample_id in range(num_samples):
        rows = random.randint(2, GRID_ROWS)
        cols = random.randint(2, GRID_COLS)
        panels = generate_valid_panels(rows, cols)
        coords_list, sequence = calculate_reading_order(panels)
        save_sample(sample_id, coords_list, sequence)

if __name__ == "__main__":
    generate_dataset(1000)
    print(f"Dataset {VERSION} generated in {DATASET_DIR}")
