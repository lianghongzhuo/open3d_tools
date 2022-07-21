import open3d as o3d
import numpy as np


def show_points(points: np.array):
    pcd = numpy2o3d(points)
    o3d.visualization.draw_geometries([pcd])


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


def create_plane(points: np.array = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])):
    """
    p0--------p1(y)
    |          |
    |          |
    p3(x)------p2
    """
    faces = np.array([[0, 3, 1], [1, 3, 2]])
    return convert_numpy_to_mesh(points, faces)


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


if __name__ == "__main__":
    cap = create_capsule()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=np.array([0, 0, 0]))
    o3d.visualization.draw_geometries([cap, mesh_frame])
    print("created capsule", cap)
