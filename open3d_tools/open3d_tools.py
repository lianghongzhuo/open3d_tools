import open3d as o3d
import numpy as np
import random
from utils import dir_name, get_rgb_colors


def show_points(points=None, points_color=None, key_points=None, key_points_color=None, key_points_colors=None,
                key_points_radius=0.01, show_norm=False, frame_size=None, frame_position=np.array([0, 0, 0]),
                show_obj_paths=None, paint_color=False, mesh_scale=None):
    """
    show points in open3d
    Args:
        points: numpy (n * 3)
        points_color: uniform color of points
        key_points: show key points using primitive shape
        key_points_color: color for keypoint
        key_points_radius: radius of key points
        show_norm: bool show norm or not
        frame_size:
        frame_position:
        show_obj_paths: mesh obj paths list to show
        paint_color: paint the mesh or not
        show_objs: show objs list
    Returns:
        show the input pointcloud in a pop out window
    """
    geo_list = []
    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if show_norm:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if points_color is not None:
            pcd.paint_uniform_color(points_color)
        geo_list.append(pcd)
    if key_points is not None:
        assert len(key_points.shape) == 2
        assert key_points.shape[1] == 3
        rgb_colors = list(get_rgb_colors().values())
        random.shuffle(rgb_colors)
        if key_points_colors is None:
            if key_points_color is None:
                key_points_color = rgb_colors[0]
            key_points_colors = [key_points_color] * len(key_points)

        for i, key_point in enumerate(key_points):
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=key_points_radius)
            trans = np.eye(4)
            trans[:3, 3] = key_point
            mesh_sphere.transform(trans)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color(key_points_colors[i])
            geo_list.append(mesh_sphere)

    if frame_size is not None:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_position)
        geo_list.append(mesh_frame)
    if show_obj_paths is not None:
        color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i, path in enumerate(show_obj_paths):
            mesh_obj = o3d.io.read_triangle_mesh(path)
            if mesh_scale is not None:
                mesh_obj = convert_numpy_to_mesh(np.array(mesh_obj.vertices)*mesh_scale, np.array(mesh_obj.triangles))
            mesh_obj.compute_vertex_normals()
            if paint_color:
                mesh_obj.paint_uniform_color(color_list[i])
            geo_list.append(mesh_obj)
    return geo_list


def open3d_show(geo_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in geo_list:
        vis.add_geometry(geo)
    vis.get_render_option().load_from_json(f"{dir_name}/../config/render_option.json")
    vis.run()
    vis.destroy_window()


def show_rgb_points(xyz, rgb):
    """Display point cloud provided from "xyz" with colors from "rgb".

    Args:
        rgb: RGB image
        xyz: X, Y and Z images (point cloud co-ordinates)

    Returns None

    """
    xyz = np.nan_to_num(xyz).reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb / 255)

    visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)

    visualizer.get_render_option().background_color = (0, 0, 0)
    visualizer.get_render_option().point_size = 1
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.get_view_control().set_front([0, 0, -1])
    visualizer.get_view_control().set_up([0, -1, 0])

    visualizer.run()
    visualizer.destroy_window()


def numpy2o3d(points: np.array) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_capsule(height=2.0, radius=1.0):
    """
    Create a mesh of a capsule, or a cylinder with hemispheric ends.
    Parameters
    ----------
    height : float
      Center to center distance of two spheres
    radius : float
      Radius of the cylinder and hemispheres
    Returns
    ----------
    capsule :
        Capsule geometry with:
            - cylinder axis is along Z
            - one hemisphere is centered at the origin
            - other hemisphere is centered along the Z axis at height
    """
    # tol_zero: Floating point numbers smaller than this are considered zero
    # set our zero for floating point comparison to 100x
    # the resolution of float64 which works out to 1e-13
    tol_zero = np.finfo(np.float64).resolution * 100
    height = float(height)
    radius = float(radius)
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    vertices = np.asarray(sphere_mesh.vertices)
    faces = np.asanyarray(sphere_mesh.triangles)
    top = vertices[:, 1] > tol_zero
    vertices[top] += [0, height, 0]
    capsule = convert_numpy_to_mesh(vertices, faces)
    return capsule


def create_plane(frame: str|np.ndarray, scale: float = 1.0):
    """
    p0----------p1
    |            |
    |     .----- |
    |     |      |
    |     |      |
    p3----------p2
    """
    if frame == "xy":
        points = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
    elif frame == "xz":
        points = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]])
    elif frame == "yz":
        points = np.array([[0, -1, -1], [0, -1, 1], [0, 1, 1], [0, 1, -1]])
    elif type(frame) is np.ndarray:
        points = frame
    else:
        raise NotImplementedError
    faces = np.array([[0, 3, 1], [1, 3, 2]])
    return convert_numpy_to_mesh(points * scale, faces)


def convert_numpy_to_mesh(obj_verts, obj_faces, color=None, compute_norm=True):
    """
    convert numpy arrays to open3d mesh
    """
    obj = o3d.geometry.TriangleMesh()
    obj.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj.vertices = o3d.utility.Vector3dVector(obj_verts)
    if color is not None:
        obj.paint_uniform_color(color)
    if compute_norm:
        obj.compute_vertex_normals()
    return obj


def convert_numpy_to_tetra(vertices, tetras):
    """
    vertices: nx3 array float
    tetras: nx4 array int
    return: obj, open3d tetra mesh
    """
    obj = o3d.geometry.TetraMesh()
    obj.vertices = o3d.utility.Vector3dVector(vertices)
    obj.tetras = o3d.utility.Vector4iVector(tetras)


def get_normal(points, radius=0.1, max_nn=30):
    """
    input: points is numpy array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_norm = np.asarray(pcd.normals)
    return pcd_norm


def voxelize_points(points, voxel_size, return_numpy=True):
    """
    Down sample the point cloud with a voxel
    """
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        # assume the input are o3d points format
        pcd = points
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if return_numpy:
        pcd = np.asarray(pcd.points)
    return pcd


if __name__ == "__main__":
    cap = create_capsule()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=np.array([0, 0, 0]))
    o3d.visualization.draw_geometries([cap, mesh_frame])
    print("created capsule", cap)
