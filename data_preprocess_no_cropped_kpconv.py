import numpy as np
import open3d as o3d
import laspy
import os
import json
import argparse
import geopandas as gpd
from shapely.geometry import box, shape, Polygon, Point
import time, csv, sys

def read_point_cloud(pc_path, ann_path):
    pc = laspy.read(pc_path)
    points = np.vstack((pc.x, pc.y, pc.z)).T
    colors = np.vstack((pc.red, pc.green, pc.blue)).T

    max_color_value = colors.max()
    if max_color_value > 255:
        colors = colors / 65280.0
    elif max_color_value > 1.5:
        colors = colors / 255.0
    else:
        colors = np.clip(colors, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    annotations = gpd.read_file(ann_path)
    return pcd, annotations

def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True

def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def normalize_pc(points):
    """
    Normalize a point cloud to fit within a unit sphere centered at the origin.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) representing the point cloud.

    Returns:
        numpy.ndarray: Normalized point cloud.
    """
    if len(points) == 0:
        return points  # Return as-is if empty

    # Compute the centroid
    centroid = np.mean(points, axis=0)

    # Center the point cloud
    points -= centroid

    # Compute the furthest distance
    furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))

    # Scale the points to fit within a unit sphere
    if furthest_distance > 0:
        points /= furthest_distance

    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for data preprocessing")
    parser.add_argument('raw', type=str, help='Path to raw data')
    parser.add_argument('out', type=str, help='Path to output folder')
    args = parser.parse_args()

    site_list = ['bagnatica', 'chiuduno', 'desio', 'euryale', 'martinengo', 'medousa', 'offanengo', 'rogno', 'stradella']

    # Global directory for saving annotated and normalized point clouds
    global_annotated_dir = os.path.join(args.out, "AnnotatedPointClouds")
    os.makedirs(global_annotated_dir, exist_ok=True)

    for site in site_list:
        print(f"Processing site {site}")

        # 1) Read entire point cloud and annotations
        pc_path = os.path.join(args.raw, site + "_point_cloud.las")
        ann_path = os.path.join(args.raw, site + "_merged_annots_georef.geojson")
        point_cloud, annotations = read_point_cloud(pc_path, ann_path)

        # 2) Process the entire point cloud for annotation
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)

        # Extract polygons and labels from annotations
        polygon_list = [Polygon(shape(row.geometry)) for _, row in annotations.iterrows()]
        polygon_labels = [row["waste_code"] for _, row in annotations.iterrows()]

        # Initialize an array for point classes (0 = background, 1 = waste)
        point_classes = np.zeros(points.shape[0], dtype=np.int32)

        # Assign labels to each point based on polygon containment
        print("Annotating points...")
        for i, point in enumerate(points):
            point_2d = Point(point[0:2])  # Use X, Y for checking polygon containment
            for poly, label in zip(polygon_list, polygon_labels):
                if poly.contains(point_2d):
                    point_classes[i] = 1  # label TODO: assign the label to the point, now binary classification
                    break  # Assign the first matching label and stop further checks

            if i % 10000 == 0:  # Log progress for every 10k points
                print(f"Processed {i}/{len(points)} points")

        print("Annotation complete.")

        # Combine points, colors, and labels into a single array
        labels = point_classes.reshape(-1, 1)

        # 3) Normalize the point cloud
        print("Normalizing point cloud...")
        points = normalize_pc(points)

        # Save the annotated and normalized point cloud as a .ply file
        annotated_ply_path = os.path.join(global_annotated_dir, f"{site}_Annotated_Normalized.ply")

        write_ply(annotated_ply_path,
                  (points, colors, labels),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        print(f"Saved annotated and normalized point cloud: {annotated_ply_path}")
