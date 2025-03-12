import torch 
import os 
import numpy as np 
import json 
from scipy.spatial.transform import Rotation

dt = {
    "transform": [
        [
            1.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0
        ]
    ],
    "scale": 1.0
}

if __name__ == "__main__":
    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/dozer2/splatfacto/2025-03-11_161130')
    checkpoint = "step-000029999.ckpt"
    
    checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

    mask = (means[:,0]>=-0.65) & (means[:,0]<=0.65) & (means[:,1]>=-1.2) & (means[:,1]<=1.2) & (means[:,2]>=-0.4) & (means[:,2]<=1.1)
    means = means[mask]
    quats = quats[mask]
    opacities = opacities[mask]
    scales = scales[mask]
    colors = colors[mask]

    # Building 
    # mask = (means[:,0]>=-0.023) & (means[:,0]<=0.087) & (means[:,1]>=-0.153) & (means[:,1]<=-0.096) & (means[:,2]>=-0.1725) & (means[:,2]<=-0.06)
    # means = means[~mask]
    # quats = quats[~mask]
    # opacities = opacities[~mask]
    # scales = scales[~mask]
    # colors = colors[~mask]

    # Tree1 
    # mask = (means[:,0]>=0.476) & (means[:,0]<=0.508) & (means[:,1]>=-0.1) & (means[:,1]<=-0.061) & (means[:,2]>=-0.162) & (means[:,2]<=-0.06)
    # means = means[~mask]
    # quats = quats[~mask]
    # opacities = opacities[~mask]
    # scales = scales[~mask]
    # colors = colors[~mask]

    # Tree2
    # mask = (means[:,0]>=0.6) & (means[:,0]<=0.64) & (means[:,1]>=-0.1) & (means[:,1]<=-0.061) & (means[:,2]>=-0.158) & (means[:,2]<=-0.055)
    # means = means[~mask]
    # quats = quats[~mask]
    # opacities = opacities[~mask]
    # scales = scales[~mask]
    # colors = colors[~mask]

    import pyvista as pv 
    means = means.detach().cpu().numpy()
    colors = colors.cpu().detach().numpy()

    N = means.shape[0]

    ply_file = f"\
ply\n\
format ascii 1.0\n\
element vertex {N}\n\
property float x\n\
property float y\n\
property float z\n\
property uint8 red\n\
property uint8 green\n\
property uint8 blue\n\
end_header\n\
"
    for i in range(N):
        ply_file += f"{means[i,0]} {means[i,1]} {means[i,2]} {np.uint8(colors[i,0]*255)} {np.uint8(colors[i,1]*255)} {np.uint8(colors[i,2]*255)}\n"

    with open('sparse_pc.ply', 'w+') as f:
        f.write(ply_file)

    plotter = pv.Plotter()

    # Create a point at the origin and add it as a mesh
    origin = np.array([[0, 0, 0]])
    point_mesh = pv.PolyData(origin)
    plotter.add_mesh(point_mesh, color='white', point_size=10, render_points_as_spheres=True)

    # Create lines for the axes:
    # X-axis from (-1, 0, 0) to (1, 0, 0)
    x_axis = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
    # Y-axis from (0, -1, 0) to (0, 1, 0)
    y_axis = pv.Line(pointa=(0, 0, 0), pointb=(0, 1, 0))
    # Z-axis from (0, 0, -1) to (0, 0, 1)
    z_axis = pv.Line(pointa=(0, 0, 0), pointb=(0, 0, 1))

    # Add the axis lines to the plotter with distinct colors
    plotter.add_mesh(x_axis, color='red', line_width=4, label='X Axis')
    plotter.add_mesh(y_axis, color='green', line_width=4, label='Y Axis')
    plotter.add_mesh(z_axis, color='blue', line_width=4, label='Z Axis')

    plotter.add_points(
        means,
        scalars=colors,   # Your RGB array
        rgb=True, 
    )

    def pick_callback(picked_mesh, point_id):
        """
        This function is called whenever a user clicks on a point
        in the rendering window. 'point' is the [x, y, z] coordinate
        of the picked point.
        """
        x, y, z = picked_mesh.points[point_id]
        print(f"Picked point coords = ({x:.3f}, {y:.3f}, {z:.3f}), ID = {point_id}")

    # Enable point picking with a callback
    plotter.enable_point_picking(
        callback=pick_callback,  # function to call on pick
        show_message=True,       # displays a hint on how to pick points
        use_mesh=True            # indicates we want to pick from the added point set
    )
    plotter.set_background('black')


    bottom_left = [-0.65, -1.2, -0.4]    # Example: (xmin, ymin, zmin)
    top_right   = [0.65, 1.2, 1.1]    # Example: (xmax, ymax, zmax)

    bounds = [bottom_left[0], top_right[0],
        bottom_left[1], top_right[1],
        bottom_left[2], top_right[2]]

    cuboid = pv.Box(bounds=bounds)

    # Add the cuboid with a transparent fill
    plotter.add_mesh(cuboid, color='cyan', opacity=0.2)

    # Overlay the edges using a wireframe style for clear visualization
    plotter.add_mesh(cuboid, color='black', style='wireframe', line_width=2)

    with open('../../nerfstudio/data/dozer2/transforms.json', 'r') as f:
        data_transform = json.load(f)

    default_forward = np.array([0, 0, -1])

    for frame in data_transform['frames']:
        mat = np.array(frame['transform_matrix'])
        # Extract the camera position from the transformation matrix
        position = mat[:3, 3]
        
        # Extract the rotation matrix (upper-left 3x3 block)
        rotation = mat[:3, :3]
        
        # Compute the camera's forward direction in world coordinates
        # If your convention differs, adjust the default_forward vector accordingly.
        forward_world = rotation.dot(default_forward)
        
        # Create a point at the camera position
        camera_point = pv.PolyData(np.array([position]))
        plotter.add_mesh(camera_point, color='magenta', point_size=10, render_points_as_spheres=True)
        
        # Create an arrow starting at the camera position pointing in the forward direction.
        # The length of the arrow can be adjusted (here set to 0.5)
        arrow = pv.Arrow(start=position, direction=forward_world, tip_length=0.2, tip_radius=0.05, shaft_radius=0.02)
        plotter.add_mesh(arrow, color='magenta')

    mat = np.array([
        [1,0,0,0],
        [0, 0.2,-0.97979589711 ,-3.7],
        [0, 0.97979589711 , 0.2, 0.8],
        [0,0,0,1]
    ])

    rpy = Rotation.from_matrix(mat[:3,:3]).as_euler('xyz')
    pos = mat[:3,3]

    for i in range(0,200):
        ori = i*np.pi*2/200
        new_ori = np.array([rpy[0], rpy[1], ori])
        new_ori_mat = Rotation.from_euler('xyz',new_ori).as_matrix()
        new_pos = np.array([3.7*np.sin(ori), -3.7*np.cos(ori), 0.8])
        new_mat = np.zeros((4,4))
        new_mat[:3,:3] = new_ori_mat 
        new_mat[:3,3] = new_pos 
        new_mat[3,3] = 1

        position = new_pos 
        rotation = new_ori_mat 
        # Compute the camera's forward direction in world coordinates
        # If your convention differs, adjust the default_forward vector accordingly.
        forward_world = rotation.dot(default_forward)
        
        # Create a point at the camera position
        camera_point = pv.PolyData(np.array([position]))
        plotter.add_mesh(camera_point, color='green', point_size=10, render_points_as_spheres=True)
        
        # Create an arrow starting at the camera position pointing in the forward direction.
        # The length of the arrow can be adjusted (here set to 0.5)
        arrow = pv.Arrow(start=position, direction=forward_world, tip_length=0.2, tip_radius=0.05, shaft_radius=0.02)
        plotter.add_mesh(arrow, color='green')
    # # Extract the camera position from the transformation matrix
    # position = mat[:3, 3]
    
    # # Extract the rotation matrix (upper-left 3x3 block)
    # rotation = mat[:3, :3]
    
    # # Compute the camera's forward direction in world coordinates
    # # If your convention differs, adjust the default_forward vector accordingly.
    # forward_world = rotation.dot(default_forward)
    
    # # Create a point at the camera position
    # camera_point = pv.PolyData(np.array([position]))
    # plotter.add_mesh(camera_point, color='blue', point_size=10, render_points_as_spheres=True)
    
    # # Create an arrow starting at the camera position pointing in the forward direction.
    # # The length of the arrow can be adjusted (here set to 0.5)
    # arrow = pv.Arrow(start=position, direction=forward_world, tip_length=0.2, tip_radius=0.05, shaft_radius=0.02)
    # plotter.add_mesh(arrow, color='blue')


    plotter.show()