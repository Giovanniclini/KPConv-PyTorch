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

def crop_point_cloud_and_polygons_with_overlap(point_cloud, annotations, grid_size, overlap_fraction=0.15):
    import numpy as np
    import open3d as o3d
    from shapely.geometry import box

    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)

    # Compute step size with overlap
    step_size = grid_size * (1 - overlap_fraction)

    # Generate the steps
    x_steps = np.arange(min_bounds[0], max_bounds[0], step_size)
    y_steps = np.arange(min_bounds[1], max_bounds[1], step_size)

    sub_clouds = {}

    for x_min in x_steps:
        for y_min in y_steps:
            # Define the bounds for the current grid cell
            x_max = x_min + grid_size
            y_max = y_min + grid_size
            min_bound = np.array([x_min, y_min, min_bounds[2]])
            max_bound = np.array([x_max, y_max, max_bounds[2]])

            # Crop the point cloud using the bounding box
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            cropped_cloud = point_cloud.crop(bounding_box)

            if np.asarray(cropped_cloud.points).shape[0] > 0:
                # Create a bounding box for the current grid cell
                cell_bbox = box(x_min, y_min, x_max, y_max)

                # Find intersecting annotations
                intersecting_annotations = annotations[annotations.intersects(cell_bbox)]
                intersecting_polygons = []

                # Extract intersecting polygons
                for _, row in intersecting_annotations.iterrows():
                    geojson_coordinates = [
                        [list(coord) for coord in row.geometry.exterior.coords]
                    ]
                    polygon_info = {
                        "type": "Polygon",
                        "coordinates": geojson_coordinates,
                        "properties": {
                            "waste_code": row["waste_code"]
                        }
                    }
                    intersecting_polygons.append(polygon_info)

                # Store the cropped cloud and polygons
                sub_clouds[(x_min, y_min)] = {
                    "point_cloud": cropped_cloud,
                    "polygons": intersecting_polygons
                }

    return sub_clouds

def filter_point_clouds_by_area_ratio(sub_clouds, grid_size, area_ratio_threshold=0.1):
    filtered_sub_clouds = {}
    bounding_box_area = grid_size ** 2

    for grid_idx, data in sub_clouds.items():
        sub_cloud = data["point_cloud"]
        polygons = data["polygons"]

        total_polygon_area = sum(
            shape({"type": "Polygon", "coordinates": poly["coordinates"]}).area
            for poly in polygons
        )

        area_ratio = total_polygon_area / bounding_box_area

        if area_ratio >= area_ratio_threshold:
            filtered_sub_clouds[grid_idx] = {
                "point_cloud": sub_cloud,
                "polygons": polygons
            }
            print(f"Sub-point cloud at {grid_idx} passed with area ratio: {area_ratio:.2f} and contains {len(np.asarray(sub_cloud.points))} points.")

    return filtered_sub_clouds

def save_filtered_sub_clouds(filtered_sub_clouds, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for grid_idx, data in filtered_sub_clouds.items():
        sub_cloud = data["point_cloud"]
        polygons = data["polygons"]

        # Save the point cloud as a .ply file
        cloud_file_path = os.path.join(output_dir, f"sub_cloud_{grid_idx[0]}_{grid_idx[1]}.ply")
        o3d.io.write_point_cloud(cloud_file_path, sub_cloud)

        # Prepare polygon data in GeoJSON format with each polygon as a Feature
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": poly["coordinates"]
                },
                "properties": poly["properties"]
            }
            for poly in polygons
        ]

        # Save the polygons as a .geojson file
        json_file_path = os.path.join(output_dir, f"sub_cloud_{grid_idx[0]}_{grid_idx[1]}.geojson")
        with open(json_file_path, "w") as json_file:
            json.dump({"type": "FeatureCollection", "features": features}, json_file, indent=4)

        print(f"Saved {cloud_file_path} and {json_file_path}")

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


def annotations_handling(output_cropped):

    for file in os.listdir(output_cropped):
        if file.endswith(".ply"):
            base_file = file.rsplit(".ply", 1)[0]
            generate_annotated_ply(base_file, output_cropped)

def generate_annotated_ply(file, output_cropped):

    base_path = os.path.join(output_cropped, file)

    # read annotations georeferenced
    with open(base_path + ".geojson") as f:
        ann_ortho_georef = json.load(f)

    poly_num = len(ann_ortho_georef["features"])
    ids = np.array([el for el in range(0, poly_num)])

    # retrieve polygons and labels
    categories = np.array([annotations["properties"]["waste_code"] for annotations in ann_ortho_georef["features"]], dtype=object)
    polygons = np.array([Polygon(polygons["geometry"]["coordinates"][0]) for polygons in ann_ortho_georef["features"]], dtype=object)

    # read the point cloud in .ply format
    point_cloud = o3d.io.read_point_cloud(base_path + ".ply")
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # associate points to segmentation polygons
    point_classes = np.zeros(points.shape[0], dtype=np.int32)
    log_interval = 15
    last_log_time = time.time()

    # Loop through points
    for i, point in enumerate(points):
        point_2d = point[0:2]
        point_geom = Point(point_2d)

        for id, poly in zip(ids, polygons):
            if poly.contains(point_geom):
                point_classes[i] = 1  # Mark as waste

        # Check if it's time to log
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            print(f"Processed {i+1} points out of {len(points)}")
            last_log_time = current_time

    print("All points have been processed.")

    # Combine points, colors, and labels into a single array
    labels = point_classes.reshape(-1, 1)

    # Save the annotated point cloud as a .ply file
    annotated_ply_dir = os.path.join(output_cropped)
    os.makedirs(annotated_ply_dir, exist_ok=True)
    annotated_ply_path = os.path.join(annotated_ply_dir, file + "Annotated.ply")

    # Use the write_ply function to save the data
    write_ply(annotated_ply_path,
            (points, colors, labels),
            ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    print(f"Saved {annotated_ply_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for data preprocessing")
    parser.add_argument('raw', type=str, help='Path to raw data')
    parser.add_argument('out', type=str, help='Path to output folder')
    args = parser.parse_args()

    site_list = ['bagnatica', 'chiuduno', 'desio', 'euryale', 'martinengo', 'medousa', 'offanengo', 'rogno', 'stradella', 'whitestone']

    # size of the grid to use to divide point cloud. expressed in metres since coordinates
    # of pcs are metre based.
    grid_size = 10

    area_ratio_threshold = 0.10

    for site in site_list:

        print("Preprocessing site ", site)

        # 1) Read entire point cloud and annotations
        pc_path = os.path.join(args.raw, site + "_point_cloud.las")
        ann_path = os.path.join(args.raw, site + "_merged_annots_georef.geojson")
        point_cloud, annotations = read_point_cloud(pc_path, ann_path)

        # 2) Crop pcs
        sub_clouds = crop_point_cloud_and_polygons_with_overlap(point_cloud, annotations, grid_size)

        # 3) Filter point clouds based on area ratio
        filtered_sub_clouds = filter_point_clouds_by_area_ratio(sub_clouds, grid_size, area_ratio_threshold)

        # 4) Save Point Clouds with associated segmentation polygons
        output_path = os.path.join(args.out, site + "Cropped")
        save_filtered_sub_clouds(filtered_sub_clouds, output_path)

        # 5) Generate labels
        annotations_handling(output_path)