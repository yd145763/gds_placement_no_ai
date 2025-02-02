# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 07:29:44 2025

@author: limyu
"""

import numpy as np
import pya
import random
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

gds_link = "G:\\My Drive\\GDS_placement\\demo\\before_arrangement.gds"
layout = pya.Layout()
layout.read(gds_link)


all_cells = []
for cell in layout.each_cell():
    cell_name = cell.name
    all_cells.append(cell_name)
all_cells.remove('TOP')


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




random_list = random.sample(range(len(all_cells)), len(all_cells))
#random_list = np.arange(0, len(all_cells), 0)

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

areas = []

for p in polygons:
    areas.append(p.area)
    
sorted_data = sorted(zip(areas, cell_names, polygons), reverse=True)

areas, cell_names, polygons = zip(*sorted_data)

step = 50000

from shapely.affinity import translate, rotate
from shapely.geometry.polygon import orient

# Check if two polygons overlap
def can_place(base_polygon, new_polygon, x_offset, y_offset, rotation_angle):
    rotated_polygon = rotate(orient(new_polygon, sign=1.0), rotation_angle, origin='centroid')
    moved_polygon = translate(rotated_polygon, xoff=x_offset, yoff=y_offset)
    return not base_polygon.intersects(moved_polygon)

def greedy_polygon_packing(polygons, bin_width, bin_height, step, margin):
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
                for angle in [0,90, 180, 270]:  # Always counterclockwise
                    rotated_polygon = rotate(polygon, angle, origin='centroid')
                    moved_polygon = translate(rotated_polygon, xoff=x_offset, yoff=y_offset)

                    # Create a buffered version to enforce margin
                    buffered_polygon = moved_polygon.buffer(margin)

                    # Print attempt details
                    print(f"  Attempt: x_offset={x_offset}, y_offset={y_offset}, rotation_angle={angle}")

                    if bin_area.contains(moved_polygon) and all(not p.buffer(margin).intersects(buffered_polygon) for p in current_bin):
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


arranged_link = "G:\\My Drive\\GDS_placement\\demo\\after_arrangement.gds"
to_delete = list(set(all_cells) - set(cell_names))
import shutil
import os

# Original file path
source_file = gds_link

# Generate a new name (e.g., file_backup.txt)
file_name, file_extension = os.path.splitext(source_file)


# Copy the file
shutil.copy(source_file, arranged_link)

# Load the layout
layout1 = pya.Layout()
layout1.read(gds_link)  # Load your GDS file

for name in to_delete:

    # Find the cell to delete
    cell_name = name  # Replace with your cell name
    cell = layout.cell(cell_name)
    
    if cell:
        layout.delete_cells([cell.cell_index()])
        print(f"Deleted cell: {cell_name}")
    else:
        print(f"Cell {cell_name} not found")
    
    # Save the updated GDS file
    layout.write(arranged_link)

# Example usage
bins, displacements, cell_names_placed = greedy_polygon_packing(polygons, 3000000, 5000000, 50000, 0)
plot_bins([polygons], 3000000, 5000000)
plot_bins(bins, 3000000, 5000000)

moved_polygons = bins[0]

recorded_angles = []
for i in range(len(all_cells)):
    displacement = displacements[i]
    recorded_angle = displacement[-1]
    recorded_angles.append(recorded_angle)


gds_polygons = []
layout1 = pya.Layout()
layout1.read(arranged_link)

I = range(len(cell_names))
for i, name, angle in zip(I, cell_names, recorded_angles):

    # Get the target cell
    cell_name = name  # Replace with the actual cell name
    cell = layout1.cell(cell_name)
    layer_to_extract = pya.LayerInfo(1, 0)
    
    if cell:
        
        if angle == 90:  
            # Define a 90-degree rotation
            rotation = pya.Trans(pya.Trans.R90, 0, 0)
    

        
            # Transform all shapes inside the cell
            for layer_index in cell.layout().layer_indices():
                shapes = cell.shapes(layer_index)
                for shape in shapes.each():
                    shape.transform(rotation)
            layout1.write(arranged_link)
        if angle == 180:  
            # Define a 180-degree rotation
            rotation = pya.Trans(pya.Trans.R180, 0, 0)
    

        
            # Transform all shapes inside the cell
            for layer_index in cell.layout().layer_indices():
                shapes = cell.shapes(layer_index)
                for shape in shapes.each():
                    shape.transform(rotation)
            layout1.write(arranged_link)
        if angle == 270:  
            # Define a 270-degree rotation
            rotation = pya.Trans(pya.Trans.R270, 0, 0)
    

        
            # Transform all shapes inside the cell
            for layer_index in cell.layout().layer_indices():
                shapes = cell.shapes(layer_index)
                for shape in shapes.each():
                    shape.transform(rotation)
            layout1.write(arranged_link)

        

        shapes = get_shapes_in_layer(cell, layer_to_extract)
        print(f"Found {len(shapes)} shapes in layer {layer_to_extract.layer}/{layer_to_extract.datatype} in cell '{cell}'.")
        for shape in shapes:
            print(shape)
        coordinates = shape_to_tuples(shape)
        gds_polygon = [(x, y) for x, y in coordinates]
        gds_polygons.append(gds_polygon)
        
    else:
        print(f"Cell '{target_cell_name}' not found.")


    gds_polygon = Polygon(gds_polygon)
    moved_polygon = moved_polygons[i]
    moved_polygon = Polygon(moved_polygon)
    
    
    
    # Extract the exterior coordinates of each polygon
    x1, y1 = gds_polygon.exterior.xy
    x2, y2 = moved_polygon.exterior.xy
    
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    
    
    # Add labels and legend
    ax.set_title('before move')
    ax.legend(['from gds', 'in theory'])
    
    # Show the plot
    plt.show()
    plt.close()
    
    total_points_gds = len(gds_polygon.exterior.coords)

    x_gds_list = []
    y_gds_list = []
    for i in range(total_points_gds):
        from_gds_point = gds_polygon.exterior.coords[i]
        x_gds, y_gds = from_gds_point[0], from_gds_point[1] 
        x_gds_list.append(x_gds)
        y_gds_list.append(y_gds)

    min_gds = min(x_gds_list)
    min_gds_x_indices = [index for index, value in enumerate(x_gds_list) if value == min_gds]
    if len(min_gds_x_indices)> 1:
        first_index = min_gds_x_indices[0]
        second_index = min_gds_x_indices[1]
    
        first_y = y_gds_list[first_index]
        second_y = y_gds_list[second_index]
    
        if first_y < second_y:
            gds_index = first_index
        else:
            gds_index = second_index
    else:
        gds_index = min_gds_x_indices[0]
    
    max_gds = max(x_gds_list)
    max_gds_x_indices = [index for index, value in enumerate(x_gds_list) if value == max_gds]
    if len(max_gds_x_indices)> 1:
        first_index = max_gds_x_indices[0]
        second_index = max_gds_x_indices[1]
    
        first_y = y_gds_list[first_index]
        second_y = y_gds_list[second_index]
    
        if first_y > second_y:
            gds_index_right = first_index
        else:
            gds_index_right = second_index
    else:
        gds_index_right = max_gds_x_indices[0]




    total_points_target = len(moved_polygon.exterior.coords)

    x_target_list = []
    y_target_list = []
    for i in range(total_points_target):
        from_target_point = moved_polygon.exterior.coords[i]
        x_target, y_target = from_target_point[0], from_target_point[1] 
        x_target_list.append(x_target)
        y_target_list.append(y_target)

    min_target = min(x_target_list)
    min_target_x_indices = [index for index, value in enumerate(x_target_list) if value == min_target]
    if len(min_target_x_indices)> 1:
        first_index = min_target_x_indices[0]
        second_index = min_target_x_indices[1]
    
        first_y = y_target_list[first_index]
        second_y = y_target_list[second_index]
    
        if first_y < second_y:
            target_index = first_index
        else:
            target_index = second_index
    else:
        target_index = min_target_x_indices[0]

    max_target = max(x_target_list)
    max_target_x_indices = [index for index, value in enumerate(x_target_list) if value == max_target]
    if len(max_target_x_indices)> 1:
        first_index = max_target_x_indices[0]
        second_index = max_target_x_indices[1]
    
        first_y = y_target_list[first_index]
        second_y = y_target_list[second_index]
    
        if first_y > second_y:
            target_index_right = first_index
        else:
            target_index_right = second_index
    else:
        target_index_right = max_target_x_indices[0]
        

    lower_left_gds = gds_polygon.exterior.coords[gds_index]
    lower_left_target = moved_polygon.exterior.coords[target_index]

    upper_right_gds = gds_polygon.exterior.coords[gds_index_right]
    upper_right_target = moved_polygon.exterior.coords[target_index_right]

    x_distance = lower_left_target[0] - lower_left_gds[0]
    x_distance = round(x_distance, 0)
    y_distance = lower_left_target[1] - lower_left_gds[1]
    y_distance = round(y_distance, 0)


    x_distance_right = upper_right_target[0] - upper_right_gds[0]
    x_distance_right = round(x_distance_right, 0)
    y_distance_right = upper_right_target[1] - upper_right_gds[1]
    y_distance_right = round(y_distance_right, 0)

    
    # Define a x y translation
    translation = pya.Trans(pya.Trans.R0, x_distance, y_distance)



    # Transform all shapes inside the cell
    for layer_index in cell.layout().layer_indices():
        shapes = cell.shapes(layer_index)
        for shape in shapes.each():
            shape.transform(translation)
    layout1.write(arranged_link)
    
    shapes = get_shapes_in_layer(cell, layer_to_extract)
    print(f"Found {len(shapes)} shapes in layer {layer_to_extract.layer}/{layer_to_extract.datatype} in cell '{cell}'.")
    for shape in shapes:
        print(shape)
    coordinates = shape_to_tuples(shape)
    gds_polygon_final = [(x, y) for x, y in coordinates]
    
    gds_polygon_final = Polygon(gds_polygon_final)
    
    # Extract the exterior coordinates of each polygon
    x1, y1 = gds_polygon_final.exterior.xy
    x2, y2 = moved_polygon.exterior.xy
    
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    
    
    # Add labels and legend
    ax.set_title('after move')
    ax.legend(['from gds', 'in theory'])
    
    # Show the plot
    plt.show()
    plt.close()
    
    
import pya

# Load the GDS file
layout1 = pya.Layout()
layout1.read(arranged_link)  # Replace with your actual GDS file

# Get all cell names
all_cells_arranged = [cell.name for cell in layout1.each_cell()]

# Dictionary to store bounding box coordinates
cell_bounding_boxes = {}

for cell_name in all_cells_arranged:
    cell = layout1.cell(cell_name)
    bbox = cell.bbox()  # Get the bounding box of the cell
    
    lower_left = (bbox.left, bbox.bottom)   # Lower-left coordinates
    upper_right = (bbox.right, bbox.top)   # Upper-right coordinates
    
    cell_bounding_boxes[cell_name] = (lower_left, upper_right)

# Print results
for cell_name, (ll, ur) in cell_bounding_boxes.items():
    print(f"Cell: {cell_name}, Lower-Left: {ll}, Upper-Right: {ur}")

lower_left_x_coords = [bbox[0][0] for bbox in cell_bounding_boxes.values()]
# Find the smallest y-coordinate and its index
min_x = min(lower_left_x_coords)
min_index = lower_left_x_coords.index(min_x)

# Find the corresponding cell name
cell_name_with_min_x = list(cell_bounding_boxes.keys())[min_index]

print(f"Smallest x: {min_x}, Found in cell: {cell_name_with_min_x}")

lower_left_y_coords = [bbox[0][1] for bbox in cell_bounding_boxes.values()]
# Find the smallest y-coordinate and its index
min_y = min(lower_left_y_coords)
min_index = lower_left_y_coords.index(min_y)

# Find the corresponding cell name
cell_name_with_min_y = list(cell_bounding_boxes.keys())[min_index]

print(f"Smallest y: {min_y}, Found in cell: {cell_name_with_min_y}")



upper_right_x_coords = [bbox[1][0] for bbox in cell_bounding_boxes.values()]
# Find the smallest y-coordinate and its index
max_x = max(upper_right_x_coords)
max_index = upper_right_x_coords.index(max_x)

# Find the corresponding cell name
cell_name_with_max_x = list(cell_bounding_boxes.keys())[max_index]

print(f"Largest x: {max_x}, Found in cell: {cell_name_with_max_x}")

upper_right_y_coords = [bbox[1][1] for bbox in cell_bounding_boxes.values()]
# Find the smallest y-coordinate and its index
max_y = max(upper_right_y_coords)
max_index = upper_right_y_coords.index(max_y)

# Find the corresponding cell name
cell_name_with_max_y = list(cell_bounding_boxes.keys())[max_index]

print(f"Largest y: {max_y}, Found in cell: {cell_name_with_max_y}")


canvas = (max_y - min_y)*(max_x - min_x)
areas_list = [i for i in areas]
ratio = (sum(areas_list))/ canvas