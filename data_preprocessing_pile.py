import os
import sys
import numpy as np
import laspy
import json
from shapely.geometry import Polygon, Point
from tqdm import tqdm

# Function to extract PLY headers
def header_properties(field_list, field_names):
    lines = []
    lines.append('element vertex %d' % field_list[0].shape[0])
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1
    return lines

# Function to write PLY files
def write_ply(filename, field_list, field_names, triangular_faces=None):
    field_list = list(field_list) if isinstance(field_list, (list, tuple)) else [field_list]
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print('wrong number of field names')
        return False

    if not filename.endswith('.ply'):
        filename += '.ply'

    with open(filename, 'w') as plyfile:
        header = ['ply', f'format binary_{sys.byteorder}_endian 1.0']
        header.extend(header_properties(field_list, field_names))
        if triangular_faces is not None:
            header.append(f'element face {triangular_faces.shape[0]}')
            header.append('property list uchar int vertex_indices')
        header.append('end_header')
        for line in header:
            plyfile.write(f"{line}\n")

    with open(filename, 'ab') as plyfile:
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list.append((field_names[i], field.dtype.str))
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

def process_annotations(annotation_folder, pointcloud_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith('.geojson')]
    pointcloud_files = [f for f in os.listdir(pointcloud_folder) if f.endswith('.las')]

    for ann_file in tqdm(annotation_files, desc="Processing Annotations"):
        ann_path = os.path.join(annotation_folder, ann_file)
        with open(ann_path, 'r') as f:
            annotations = json.load(f)

        # Extract polygons and category IDs
        polygons = []
        for feature in annotations["features"]:
            polygon = Polygon(feature["geometry"]["coordinates"][0])
            category_id = feature["properties"]["category_id"]
            polygons.append((polygon, category_id))

        # Find corresponding point cloud
        base_name = os.path.splitext(ann_file)[0]
        las_file = next((f for f in pointcloud_files if base_name in f), None)

        if las_file is None:
            print(f"No matching LAS file for {ann_file}, skipping.")
            continue

        las_path = os.path.join(pointcloud_folder, las_file)
        las_data = laspy.read(las_path)

        # Extract point data
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T
        colors = np.vstack((las_data.red, las_data.green, las_data.blue)).T.astype(np.uint8)  # Ensure correct dtype
        num_points = points.shape[0]

        # Process each polygon separately
        for i, (polygon, category_id) in enumerate(polygons):
            inside = np.array([polygon.contains(Point(x, y)) for x, y, _ in points])
            point_labels = np.full(num_points, 6, dtype=np.int32)  # Default to background
            point_labels[inside] = category_id  # Assign waste category inside polygon

            # Extract only points within the bounding box of this polygon
            min_x, min_y = np.min(np.array(polygon.exterior.coords)[:, :2], axis=0)
            max_x, max_y = np.max(np.array(polygon.exterior.coords)[:, :2], axis=0)

            bounding_box_mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & \
                                (points[:, 1] >= min_y) & (points[:, 1] <= max_y)

            cropped_points = points[bounding_box_mask]
            cropped_colors = colors[bounding_box_mask]
            cropped_labels = point_labels[bounding_box_mask]

            # Skip if there are no points inside
            if cropped_points.shape[0] == 0:
                continue

            # Write to PLY with RGB
            output_ply = os.path.join(output_folder, f"{base_name}_polygon_{i}.ply")
            write_ply(output_ply, (cropped_points, cropped_colors, cropped_labels), 
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            print(f"Saved: {output_ply}")

# Set folder paths
annotation_folder = "/root/filtered_annotations"
pointcloud_folder = "/root/pile_pc"
output_folder = "/root/point_cloud_pile_annotated"

process_annotations(annotation_folder, pointcloud_folder, output_folder)
