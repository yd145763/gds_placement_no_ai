# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:17:17 2025

@author: limyu
"""


import numpy as np
import pya
import random
import pandas as pd
from shapely.geometry import Polygon

gds_link = "C:\\Users\\limyu\\Google Drive\\GDS_placement\\with_base_backed.gds"
layout = pya.Layout()
layout.read(gds_link)


all_cells = []
for cell in layout.each_cell():
    cell_name = cell.name
    all_cells.append(cell_name)
all_cells.remove('TOP')
all_cells.remove('soi_8channel0')
all_cells.remove('soi_4ring0')

def movetozero(cellname):

    # Load the layout
    layout = pya.Layout()
    layout.read(gds_link)
    
    all_cells = []
    for cell in layout.each_cell():
        cell_name = cell.name
        all_cells.append(cell_name)
    all_cells.remove('TOP')
    
    meow_index = all_cells.index(cellname)
    
    
    # Get the cells (assuming you know which ones have the shapes)
    
    cell2 = layout.cell(meow_index)  # Second cell
    
    
    # Get the transformation of the cell in the layout

    instances = list(layout.top_cell().each_inst())    # Assuming top_cell is the root of your layout
    
    # Find the transformation of the target cell
    transformation = None
    for instance in instances:
        if instance.cell.name == cell2.name:
            transformation = instance.trans
    
    if transformation is not None:
        # Apply the transformation to the bounding box
        bbox = cell2.bbox()
        global_bbox = bbox.transformed(transformation)
    
        # Calculate the global center
        global_center = global_bbox.center()
    
    print('x = '+str(global_center.x), 'y = '+str(global_center.y))
        
    
    random_number_x = -1*global_center.x #in nm
    random_number_y = -1*global_center.y #in nm
    
    # Load the layout
    layout = pya.Layout()
    layout.read(gds_link)
    
    all_cells = []
    for cell in layout.each_cell():
        cell_name = cell.name
        all_cells.append(cell_name)
    #all_cells.remove('TOP')
    
    meow_index = all_cells.index(cellname)
    
    
    # Get the cells (assuming you know which ones have the shapes)
    
    cell2 = layout.cell(meow_index)  # Second cell
    
    def move_cell(cell, delta_x, delta_y):
        """
        Moves the cell by `delta_x` microns in the X direction and `delta_y` microns in the Y direction.
        This function applies the transformation to all shapes across all layers in the cell.
        """
        current_position_x = 0
        current_position_y = 0
        
        # Apply transformation relative to the current position (X and Y)
        transformation = pya.Trans(0, False, delta_x - current_position_x, delta_y - current_position_y)
    
        # Update the current position
        current_position_x = delta_x
        current_position_y = delta_y
        
        # Iterate over all possible layer indices (assuming layers are indexed starting from 0)
        for layer_index in range(cell.layout().layers()):
            # Check if the cell has shapes in the current layer
            if not cell.shapes(layer_index).is_empty():
                # Transform all shapes in this layer by the new transformation
                for shape in cell.shapes(layer_index).each():
                    shape.transform(transformation)
    
    # Example: Move the cell by `random_number_x` and `random_number_y`
    move_cell(cell2, random_number_x, random_number_y)
    
    print('moved by x = '+str(random_number_x), 'y = '+str(random_number_y))
    
    # Save the layout to see the transformation
    layout.write(gds_link)

for c in all_cells:
    movetozero(c)

# Function to get all shapes in a specific layer
def get_shapes_in_layer(cell, layer_info):
    shapes = []
    layer_index = cell.layout().find_layer(layer_info)
    if layer_index != -1:  # Check if the layer exists
        for shape in cell.shapes(layer_index):  # Iterate over shapes in the layer
            shapes.append(shape)
    else:
        print(f"Layer {layer_info.layer}/{layer_info.datatype} not found in the cell.")
    return shapes

def shape_to_tuples(shape):
    if shape.is_polygon():
        # If the shape is a polygon, extract its points
        polygon = shape.polygon
        return [(point.x, point.y) for point in polygon.each_point_hull()]
    elif shape.is_box():
        # If the shape is a box, extract its corners
        box = shape.box  # Remove parentheses, as `box` is a property
        return [
            (box.p1.x, box.p1.y),
            (box.p2.x, box.p1.y),
            (box.p2.x, box.p2.y),
            (box.p1.x, box.p2.y)
        ]
    elif shape.is_path():
        # If the shape is a path, extract its points
        path = shape.path
        return [(point.x, point.y) for point in path.each_point()]
    else:
        print("Unsupported shape type.")
        return []

import matplotlib.pyplot as plt

def plot_bins(bins, bin_width, bin_height):
    for bin_idx, bin_polygons in enumerate(bins):
        fig, ax = plt.subplots(figsize=(3, 5))
        
        # Calculate bounding box of all polygons in the bin
        if bin_polygons:
            all_x = []
            all_y = []
            for poly in bin_polygons:
                x, y = poly.exterior.xy
                all_x.extend(x)
                all_y.extend(y)
            
            ax.set_xlim(min(all_x) - 100, max(all_x) + 100)  # Add margin
            ax.set_ylim(min(all_y) - 100, max(all_y) + 100)  # Add margin
        else:
            ax.set_xlim(0, bin_width)
            ax.set_ylim(0, bin_height)
        
        ax.set_title(f"Bin {bin_idx + 1}")
        
        for poly in bin_polygons:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5)
        
        plt.show()

random_list = random.sample(range(len(all_cells)), 20)

cell_names = []
pya_shape_coordinates = []
tup_shape_coordinates = []



for r in random_list:

    # Example: Get shapes for a specific layer in a specific cell
    target_cell_name = all_cells[r]  # Replace with the desired cell name
    cell_names.append(target_cell_name)
    target_cell = layout.cell(target_cell_name)
    
    # Define the layer to extract (e.g., layer 1, datatype 0)
    layer_to_extract = pya.LayerInfo(1, 0)
    
    if target_cell:
        shapes = get_shapes_in_layer(target_cell, layer_to_extract)
        print(f"Found {len(shapes)} shapes in layer {layer_to_extract.layer}/{layer_to_extract.datatype} in cell '{target_cell_name}'.")
        for shape in shapes:
            print(shape)
    else:
        print(f"Cell '{target_cell_name}' not found.")
    
    pya_shape_coordinates.append(shape)
    # Example usage
    # Assuming `shape` is the pya.Shape object
    coordinates = shape_to_tuples(shape)
    
    coordinates = [(x, y) for x, y in coordinates]
    tup_shape_coordinates.append(coordinates)

    # Print the result
    print(coordinates)


polygons = []

for p in tup_shape_coordinates:
    polygon = Polygon(p)
    polygons.append(polygon)

original_polygons = polygons

step = 100000

from shapely.affinity import translate, rotate
from shapely.geometry.polygon import orient

# Check if two polygons overlap
def can_place(base_polygon, new_polygon, x_offset, y_offset, rotation_angle):
    rotated_polygon = rotate(orient(new_polygon, sign=1.0), rotation_angle, origin='centroid')
    moved_polygon = translate(rotated_polygon, xoff=x_offset, yoff=y_offset)
    return not base_polygon.intersects(moved_polygon)

def greedy_polygon_packing(polygons, bin_width, bin_height):
    bins = []  # List of bins, each bin is a list of placed polygons
    current_bin = []  # Current bin
    bin_area = Polygon([(0, 0), (bin_width, 0), (bin_width, bin_height), (0, bin_height)])
    displacements = []
    cell_names_placed = []

    for polygon_idx, polygon in enumerate(polygons):
        placed = False
        print(f"Placing polygon {polygon_idx + 1}...")

        # Ensure consistent polygon orientation
        polygon = orient(polygon, sign=1.0)

        # Try to place the polygon in the current bin
        for x_offset in np.arange(0, bin_width, step):
            for y_offset in np.arange(0, bin_height, step):
                for angle in [0, 90]:  # Always counterclockwise
                    rotated_polygon = rotate(polygon, angle, origin='centroid')
                    moved_polygon = translate(rotated_polygon, xoff=x_offset, yoff=y_offset)

                    # Print attempt details
                    print(f"  Attempt: x_offset={x_offset}, y_offset={y_offset}, rotation_angle={angle}")

                    if bin_area.contains(moved_polygon) and all(not p.intersects(moved_polygon) for p in current_bin):
                        print(f"  Success: Placed polygon {polygon_idx} at x_offset={x_offset}, y_offset={y_offset}, rotation_angle={angle}")
                        current_bin.append(moved_polygon)
                        displacements.append([x_offset, y_offset, angle])
                        cell_names_placed.append(cell_names[polygon_idx])
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        # If not placed, start a new bin
        if not placed:
            print(f"  Failed to place polygon {polygon_idx + 1} in current bin. Starting a new bin.")
            bins.append(current_bin)
            current_bin = [polygon]  # Place the polygon in the new bin

    # Add the last bin if it has polygons
    if current_bin:
        bins.append(current_bin)

    return bins, displacements, cell_names_placed



# Example usage
bins, displacements, cell_names_placed = greedy_polygon_packing(polygons, 3000000, 5000000)
plot_bins([polygons], 3000000, 5000000)
plot_bins(bins, 3000000, 5000000)



import shutil
import os

# Original file path
source_file = gds_link

# Generate a new name (e.g., file_backup.txt)
file_name, file_extension = os.path.splitext(source_file)
destination_file = "cleaned.gds"

# Copy the file
shutil.copy(source_file, destination_file)

print(f"Backup created: {destination_file}")

gds_copied = "cleaned.gds"

def calculate_rotation_angle(poly1, poly2):
    """
    Calculate the rotation angle (in degrees) between two polygons.
    
    Parameters:
    - poly1: Shapely Polygon (original)
    - poly2: Shapely Polygon (rotated)
    
    Returns:
    - Rotation angle in degrees
    """
    def get_orientation(points):
        """Finds the principal orientation of a polygon using PCA (SVD)."""
        points = np.array(points)
        points -= np.mean(points, axis=0)  # Center the points
        _, _, vh = np.linalg.svd(points)  # Perform SVD
        angle = np.arctan2(vh[0, 1], vh[0, 0])  # Compute the orientation angle
        return np.degrees(angle)
    
    # Extract coordinates from the polygon
    points1 = list(poly1.exterior.coords)
    points2 = list(poly2.exterior.coords)

    # Compute the principal angles
    angle1 = get_orientation(points1)
    angle2 = get_orientation(points2)
    
    # Compute rotation difference
    rotation_angle = angle2 - angle1
    
    return rotation_angle


moved_polygons = bins[0]

calculated_angles = []
recorded_angles = []
x_distances = []
y_distances = []
x_distances_right = []
y_distances_right = []

for i in range(20):
    cell_name = cell_names[i]
    from_bin = moved_polygons[i]
    from_polygon = polygons[i]
    displacement = displacements[i]
    recorded_angle = displacement[-1]
    recorded_angles.append(recorded_angle)
    angle = calculate_rotation_angle(from_polygon, from_bin)
    angle = int(round(angle, 10))
    print(f"Estimated Rotation Angle: {angle:.2f} degrees")
    calculated_angles.append(angle)
    rotated_polygon = rotate(from_polygon, recorded_angle, origin='centroid')

        

    
    total_points_bin = len(from_bin.exterior.coords)
    
    x_bin_list = []
    y_bin_list = []
    for i in range(total_points_bin):
        from_bin_point = from_bin.exterior.coords[i]
        x_bin, y_bin = from_bin_point[0], from_bin_point[1] 
        x_bin_list.append(x_bin)
        y_bin_list.append(y_bin)
    
    min_bin = min(x_bin_list)
    min_bin_x_indices = [index for index, value in enumerate(x_bin_list) if value == min_bin]
    
    first_index = min_bin_x_indices[0]
    second_index = min_bin_x_indices[1]
    
    first_y = y_bin_list[first_index]
    second_y = y_bin_list[second_index]
    
    if first_y < second_y:
        bin_index = first_index
    else:
        bin_index = second_index
    
    max_bin = max(x_bin_list)
    max_bin_x_indices = [index for index, value in enumerate(x_bin_list) if value == max_bin]
    
    first_index = max_bin_x_indices[0]
    second_index = max_bin_x_indices[1]
    
    first_y = y_bin_list[first_index]
    second_y = y_bin_list[second_index]
    
    if first_y > second_y:
        bin_index_right = first_index
    else:
        bin_index_right = second_index
    
    
    
    total_points_rotated = len(rotated_polygon.exterior.coords)
    
    x_rotated_list = []
    y_rotated_list = []
    for i in range(total_points_rotated):
        from_rotated_point = rotated_polygon.exterior.coords[i]
        x_rotated, y_rotated = from_rotated_point[0], from_rotated_point[1] 
        x_rotated_list.append(x_rotated)
        y_rotated_list.append(y_rotated)
    
    min_rotated = min(x_rotated_list)
    min_rotated_x_indices = [index for index, value in enumerate(x_rotated_list) if value == min_rotated]
    
    first_index = min_rotated_x_indices[0]
    second_index = min_rotated_x_indices[1]
    
    first_y = y_rotated_list[first_index]
    second_y = y_rotated_list[second_index]
    
    if first_y < second_y:
        rotated_index = first_index
    else:
        rotated_index = second_index
    
    max_rotated = max(x_rotated_list)
    max_rotated_x_indices = [index for index, value in enumerate(x_rotated_list) if value == max_rotated]
    
    first_index = max_rotated_x_indices[0]
    second_index = max_rotated_x_indices[1]
    
    first_y = y_rotated_list[first_index]
    second_y = y_rotated_list[second_index]
    
    if first_y > second_y:
        rotated_index_right = first_index
    else:
        rotated_index_right = second_index
        
    
    lower_left_bin = from_bin.exterior.coords[bin_index]
    lower_left_rotated = rotated_polygon.exterior.coords[rotated_index]
    
    upper_right_bin = from_bin.exterior.coords[bin_index_right]
    upper_right_rotated = rotated_polygon.exterior.coords[rotated_index_right]
    
    x_distance = lower_left_bin[0] - lower_left_rotated[0]
    x_distance = round(x_distance, 0)
    y_distance = lower_left_bin[1] - lower_left_rotated[1]
    y_distance = round(y_distance, 0)
    
    x_distance_right = upper_right_bin[0] - upper_right_rotated[0]
    x_distance_right = round(x_distance_right, 0)
    y_distance_right = upper_right_bin[1] - upper_right_rotated[1]
    y_distance_right = round(y_distance_right, 0)
    
    color = 'brown'
    if int(round(x_distance, 0)) != int(round(x_distance_right, 0)) or int(round(y_distance, 0)) != int(round(y_distance_right, 0)):
        color = 'purple'
    
    x_distances.append(x_distance)
    y_distances.append(y_distance)
    
    x_distances_right.append(x_distance_right)
    y_distances_right.append(y_distance_right)
    
    moved_polygon = translate(rotated_polygon, xoff=x_distance, yoff=y_distance)

    
    # Extract the exterior coordinates of each polygon
    x1, y1 = from_polygon.exterior.xy
    x2, y2 = from_bin.exterior.xy
    x3, y3 = rotated_polygon.exterior.xy
    x4, y4 = moved_polygon.exterior.xy
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    ax.fill(x3, y3, alpha=0.5, label='Polygon 2', color='green')
    ax.fill(x4, y4, alpha=0.5, label='Polygon 2', color=color)
    
    # Add labels and legend
    ax.set_title('Plot of Two Polygons, cell name = '+cell_name+', calculated angle = '+str(angle)+', recorded angle = '+str(recorded_angle))
    ax.legend(['Before arranged', 'After arranged', 'Rotated only', 'moved!'])
    
    # Show the plot
    plt.show()
    plt.close()

import sys
if x_distances != x_distances_right or y_distances !=y_distances_right:
    print('opps')
    sys.exit()

arranged_link = "C:\\Users\\limyu\\Google Drive\\GDS_placement\\cleaned.gds"

layout1 = pya.Layout()
layout1.read(gds_link)


all_cells1 = []
for cell in layout1.each_cell():
    cell_name = cell.name
    all_cells1.append(cell_name)
all_cells1.remove('TOP')

for name, x, y, angle in zip(cell_names, x_distances, y_distances, recorded_angles):

    # Get the target cell
    cell_name = name  # Replace with the actual cell name
    cell = layout1.cell(cell_name)
    
    if cell:
        
        if angle == 90:
            # Define a 90-degree rotation
            rotation = pya.Trans(pya.Trans.R90, 0, 0)
    
            # Define the translation (1,000,000 nm in x and 2,000,000 nm in y)
            translation = pya.Trans(pya.Trans.R0, x, y)
        
            # Combine both transformations (rotation first, then translation)
            final_transform = translation * rotation  # Order matters!
        
            # Transform all shapes inside the cell
            for layer_index in cell.layout().layer_indices():
                shapes = cell.shapes(layer_index)
                for shape in shapes.each():
                    shape.transform(final_transform)
            print(f"Transformation of {name} applied successfully! x = {x}, y = {y}, angle = {angle}")
        else:
            # Define the translation (1,000,000 nm in x and 2,000,000 nm in y)
            translation = pya.Trans(pya.Trans.R0, x, y)
            
            # Transform all shapes inside the cell
            for layer_index in cell.layout().layer_indices():
                shapes = cell.shapes(layer_index)
                for shape in shapes.each():
                    shape.transform(translation)       
            print(f"Transformation of {name} applied successfully! x = {x}, y = {y}, angle = {angle}")
        # Save the modified layout
        layout1.write(arranged_link)
    else:
        print(f"Cell '{name}' not found in the layout.")

