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
        fig, ax = plt.subplots()
        
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

random_list = [random.randint(0, len(all_cells) - 1) for _ in range(20)]

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

step = 50000

from shapely.affinity import translate, rotate

# Check if two polygons overlap
def can_place(base_polygon, new_polygon, x_offset, y_offset, rotation_angle):
    moved_polygon = translate(new_polygon, xoff=x_offset, yoff=y_offset)
    rotated_polygon = rotate(moved_polygon, rotation_angle, origin='centroid')
    return not base_polygon.intersects(rotated_polygon)

def greedy_polygon_packing(polygons, bin_width, bin_height):
    bins = []  # List of bins, each bin is a list of placed polygons
    current_bin = []  # Current bin
    bin_area = Polygon([(0, 0), (bin_width, 0), (bin_width, bin_height), (0, bin_height)])
    displacements = []
    cell_names_placed = []
    for polygon_idx, polygon in enumerate(polygons):
        placed = False
        print(f"Placing polygon {polygon_idx + step}...")
        
        # Try to place the polygon in the current bin
        for x_offset in np.arange(0, bin_width, step):
            for y_offset in range(0, bin_height, step):
                for angle in [0, 90, 180, 270]:
                    moved_polygon = translate(polygon, xoff=x_offset, yoff=y_offset)
                    rotated_polygon = rotate(moved_polygon, angle, origin='centroid')
                    
                    # Print attempt details
                    print(f"  Attempt: x_offset={x_offset}, y_offset={y_offset}, rotation_angle={angle}")
                    
                    if bin_area.contains(rotated_polygon) and all(not p.intersects(rotated_polygon) for p in current_bin):
                        print(f"  Success: Placed polygon {polygon_idx} in current bin at x_offset={x_offset}, y_offset={y_offset}, rotation_angle={angle}")
                        current_bin.append(rotated_polygon)
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
bins, displacements, cell_names_placed = greedy_polygon_packing(polygons, 2000000, 5000000)
plot_bins([polygons], 5000000, 5000000)
plot_bins(bins, 5000000, 5000000)



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
for i in range(20):
    cell_name = cell_names[i]
    from_bin = moved_polygons[i]
    from_polygon = polygons[i]
    
    angle = calculate_rotation_angle(from_polygon, from_bin)
    angle = int(round(angle, 10))
    print(f"Estimated Rotation Angle: {angle:.2f} degrees")
    calculated_angles.append(angle)
    rotated_polygon = rotate(from_polygon, angle, origin='centroid')
    if angle == 270:
        rotated_polygon = rotate(from_polygon, 180, origin='centroid')
        
    # Extract the exterior coordinates of each polygon
    x1, y1 = from_polygon.exterior.xy
    x2, y2 = from_bin.exterior.xy
    x3, y3 = rotated_polygon.exterior.xy
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    ax.fill(x3, y3, alpha=0.5, label='Polygon 2', color='green')
    
    # Add labels and legend
    ax.set_title('Plot of Two Polygons, cell name = '+cell_name+', angle = '+str(angle))
    ax.legend(['Before arranged', 'After arranged', 'Rotated only'])
    
    # Show the plot
    plt.show()
    plt.close()

    




#=========================

def move_cell1(cellname, delta_x, delta_y):
    
    # Load the layout
    layout = pya.Layout()
    layout.read(gds_copied)
    
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
        
    
    random_number_x = delta_x #in nm
    random_number_y = delta_y #in nm
    
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
    layout.write(gds_copied)


# Function to rotate the cell around its center
def rotate_cell_around_center(cellname, angle_degrees, layout):
    # Get the cell by name
    cell = layout.cell(cellname)
    
    # Get the bounding box of the cell
    bbox = cell.bbox()
    
    # Calculate the center of the bounding box
    center_x = (bbox.p1.x + bbox.p2.x) / 2
    center_y = (bbox.p1.y + bbox.p2.y) / 2
    
    # Create transformations
    to_origin = pya.Trans(-center_x, -center_y)
    rotation = pya.Trans(pya.Trans.R90 if angle_degrees == 90 else angle_degrees, False)
    back_to_center = pya.Trans(center_x, center_y)
    
    # Apply transformations to all shapes in the cell
    for layer_index in range(cell.layout().layers()):
        if not cell.shapes(layer_index).is_empty():
            for shape in cell.shapes(layer_index).each():
                shape.transform(to_origin)
                shape.transform(rotation)
                shape.transform(back_to_center)
    
    print(f"Cell rotated by {angle_degrees} degrees")
    return layout





import pya

# Function to get all points in the polygon or box in layer one from a cell
def celltopolygon(cell, layout):
    # Get the cell
    cell = layout.cell(cell)
    
    # Specify the layer index (layer 1, datatype 0)
    layer_index = layout.layer(1, 0) 
    
    # Extract all polygon and box points
    all_points = []
    layer_shapes = cell.shapes(layer_index)
    for shape in layer_shapes.each():
        if shape.is_polygon():
            # Extract points for polygons
            polygon = shape.polygon
            points = [point for point in polygon.each_point_hull()]
            all_points.append(points)
        elif shape.is_box():
            # Extract points for boxes
            box = shape.box
            points = [
                box.p1,  # Bottom-left corner
                pya.Point(box.p2.x, box.p1.y),  # Bottom-right corner
                box.p2,  # Top-right corner
                pya.Point(box.p1.x, box.p2.y)  # Top-left corner
            ]
            all_points.append(points)

    return all_points




"""


# Get the points for the initial state
all_points = celltopolygon('soi_8channel0', layout)

# Initialize the plot
polygon_point_list = []
for i, points in enumerate(all_points):
    for point in points:
        coordinate = (point.x, point.y)
        polygon_point_list.append(coordinate)

# Create a Polygon from the points
poly = Polygon(polygon_point_list)
x1, y1 = poly.exterior.xy

# Create the initial plot
fig, ax = plt.subplots()
ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
ax.set_title('Plot of Polygon')
ax.legend(['Initial'])
plt.show()
plt.close()


# Rotate and plot the polygon 10 times
for _ in range(10):
    # Rotate the cell
    layout = rotate_cell_around_center('soi_8channel0', 90, layout)
    
    # Get the points after rotation
    all_points = celltopolygon('soi_8channel0', layout)
    polygon_point_list = []
    
    for i, points in enumerate(all_points):
        for point in points:
            coordinate = (point.x, point.y)
            polygon_point_list.append(coordinate)
    
    # Create a Polygon from the points
    poly = Polygon(polygon_point_list)
    x1, y1 = poly.exterior.xy

    # Create a plot
    fig, ax = plt.subplots()
    ax.fill(x1, y1, alpha=0.5, label=f'Rotation {_ + 1}', color='blue')
    ax.set_title('Plot of Rotated Polygon')
    ax.legend([f'Rotation {_ + 1}'])
    plt.show()
    plt.close()


"""

# Load the GDS file

layout = pya.Layout()
layout.read(gds_copied)

from_bin_list = bins[0]

for i in range(20):
    cell_name = cell_names[i]
    displacement = displacements[i]
    angle = displacement[-1]
    from_bin = from_bin_list[i]
    all_points = celltopolygon(cell_names[i], layout)
    polygon_point_list = []
    
    for points in all_points:
        for point in points:
            coordinate = (point.x, point.y)
            polygon_point_list.append(coordinate)
    
    from_gds = Polygon(polygon_point_list)
    
    # Extract the exterior coordinates of each polygon
    x1, y1 = from_gds.exterior.xy
    x2, y2 = from_bin.exterior.xy
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    
    # Add labels and legend
    ax.set_title('Plot of Two Polygons, rotation = '+str(angle)+' cell name = '+cell_name)
    ax.legend(['From GDS', 'From bin'])
    
    # Show the plot
    plt.show()
    plt.close()
    
    
    #==========================================
    if angle == 0:
        repeat = 0
    if angle == 90:
        repeat = 1
    if angle == 180:
        repeat = 2
    if angle == 270:
        repeat = 3
    if angle == 360:
        repeat =0
    for _ in range(repeat):

        
        layout = pya.Layout()
        layout.read(gds_copied)
        layout = rotate_cell_around_center(cell_name, 90, layout)
        layout.write(gds_copied)
        layout = pya.Layout()
        layout.read(gds_copied)
        
        all_points = celltopolygon(cell_name, layout)
        polygon_point_list = []
        
        for points in all_points:
            for point in points:
                coordinate = (point.x, point.y)
                polygon_point_list.append(coordinate)
        
        from_gds = Polygon(polygon_point_list)
        
        # Extract the exterior coordinates of each polygon
        x1, y1 = from_gds.exterior.xy
        x2, y2 = from_bin.exterior.xy
    
    
    total_points = len(from_bin.exterior.coords)
    
    x_bin_list = []
    y_bin_list = []
    for i in range(total_points):
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
    
    total_points_gds = len(from_gds.exterior.coords)
    
    x_gds_list = []
    y_gds_list = []
    for i in range(total_points_gds):
        from_gds_point = from_gds.exterior.coords[i]
        x_gds, y_gds = from_gds_point[0], from_gds_point[1] 
        x_gds_list.append(x_gds)
        y_gds_list.append(y_gds)
    
    min_gds = min(x_gds_list)
    min_gds_x_indices = [index for index, value in enumerate(x_gds_list) if value == min_gds]
    
    first_index = min_gds_x_indices[0]
    second_index = min_gds_x_indices[1]
    
    first_y = y_gds_list[first_index]
    second_y = y_gds_list[second_index]
    
    if first_y < second_y:
        gds_index = first_index
    else:
        gds_index = second_index
    
    lower_left_bin = from_bin.exterior.coords[bin_index]
    lower_left_gds = from_gds.exterior.coords[gds_index]
    
    x_distance = lower_left_bin[0] - lower_left_gds[0]
    y_distance = lower_left_bin[1] - lower_left_gds[1]
    
    area_bin = from_bin.area
    area_gds = from_gds.area
    
    print('area bin = ', area_bin)
    print('area gds = ', area_gds)
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    
    ax.scatter(lower_left_gds[0], lower_left_gds[1], color='black', marker = 'x')
    ax.scatter(lower_left_bin[0], lower_left_bin[1], color='blue', marker = 'x')
    # Add labels and legend
    ax.set_title('Aligned Plot of Two Polygons, rotation = '+str(angle)+' cell name = '+cell_name+' cell index = '+str(i))
    ax.legend(['From GDS', 'From bin'])
    
    # Show the plot
    plt.show()
    plt.close()


"""


for meow in range(20):
    cell_index = meow
    displacement = displacements[cell_index]
    angle = displacement[-1]
    from_bin = from_bin_list[cell_index]
    cell_name = cell_names[cell_index]
    if angle == 0:
        repeat = 0
    if angle == 90:
        repeat = 1
    if angle == 180:
        repeat = 2
    if angle == 270:
        repeat = 3
    if angle == 360:
        repeat =0
    
    
    for _ in range(repeat):

        
        layout = pya.Layout()
        layout.read(gds_copied)
        layout = rotate_cell_around_center(cell_name, 90, layout)
        layout.write(gds_copied)
        layout = pya.Layout()
        layout.read(gds_copied)
        
        all_points = celltopolygon(cell_name, layout)
        polygon_point_list = []
        
        for points in all_points:
            for point in points:
                coordinate = (point.x, point.y)
                polygon_point_list.append(coordinate)
        
        from_gds = Polygon(polygon_point_list)
        
        # Extract the exterior coordinates of each polygon
        x1, y1 = from_gds.exterior.xy
        x2, y2 = from_bin.exterior.xy
    
    
    total_points = len(from_bin.exterior.coords)
    
    x_bin_list = []
    y_bin_list = []
    for i in range(total_points):
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
    
    total_points_gds = len(from_gds.exterior.coords)
    
    x_gds_list = []
    y_gds_list = []
    for i in range(total_points_gds):
        from_gds_point = from_gds.exterior.coords[i]
        x_gds, y_gds = from_gds_point[0], from_gds_point[1] 
        x_gds_list.append(x_gds)
        y_gds_list.append(y_gds)
    
    min_gds = min(x_gds_list)
    min_gds_x_indices = [index for index, value in enumerate(x_gds_list) if value == min_gds]
    
    first_index = min_gds_x_indices[0]
    second_index = min_gds_x_indices[1]
    
    first_y = y_gds_list[first_index]
    second_y = y_gds_list[second_index]
    
    if first_y < second_y:
        gds_index = first_index
    else:
        gds_index = second_index
    
    lower_left_bin = from_bin.exterior.coords[bin_index]
    lower_left_gds = from_gds.exterior.coords[gds_index]
    
    x_distance = lower_left_bin[0] - lower_left_gds[0]
    y_distance = lower_left_bin[1] - lower_left_gds[1]
    
    area_bin = from_bin.area
    area_gds = from_gds.area
    
    print('area bin = ', area_bin)
    print('area gds = ', area_gds)
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Plot the polygons
    ax.fill(x1, y1, alpha=0.5, label='Polygon 1', color='blue')
    ax.fill(x2, y2, alpha=0.5, label='Polygon 2', color='red')
    
    ax.scatter(lower_left_gds[0], lower_left_gds[1], color='black', marker = 'x')
    ax.scatter(lower_left_bin[0], lower_left_bin[1], color='blue', marker = 'x')
    # Add labels and legend
    ax.set_title('Aligned Plot of Two Polygons, rotation = '+str(angle)+' cell name = '+cell_name+' cell index = '+str(meow))
    ax.legend(['From GDS', 'From bin'])
    
    # Show the plot
    plt.show()
    plt.close()
"""