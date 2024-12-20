import pyvista as pv
import numpy as np
from numpy.linalg import inv, det

mean = np.array([-2, -2, 10])
cov = np.array([[0.16, 0.05, 0.02],
                [0.05, 0.25, 0.01],
                [0.02, 0.01, 0.5 ]])

cov_inv = inv(cov)
norm_factor = (1 / ((2*np.pi)**(3/2) * np.sqrt(det(cov))))

# Define region and grid
eigvals, _ = np.linalg.eigh(cov)
max_std = np.sqrt(np.max(eigvals))
extent = 3 * max_std
x_min, x_max = mean[0]-extent, mean[0]+extent
y_min, y_max = mean[1]-extent, mean[1]+extent
z_min, z_max = mean[2]-extent, mean[2]+extent

nx, ny, nz = 30, 30, 30
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

diff = points - mean
pdf_values = norm_factor * np.exp(-0.5 * np.sum((diff @ cov_inv) * diff, axis=1))
pdf_values_3d = pdf_values.reshape((nx, ny, nz))

# Use ImageData instead of UniformGrid
grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.origin = (x_min, y_min, z_min)
grid.spacing = ((x_max - x_min)/(nx-1), (y_max - y_min)/(ny-1), (z_max - z_min)/(nz-1))
grid['pdf'] = pdf_values_3d.flatten(order='C')

# Visualize
plotter = pv.Plotter()
x_line = pv.Line((0,0,0), (5,0,0))
y_line = pv.Line((0,0,0), (0,5,0))
z_line = pv.Line((0,0,0), (0,0,5))
plotter.add_mesh(x_line, color='red', line_width=5)
plotter.add_mesh(y_line, color='green', line_width=5)
plotter.add_mesh(z_line, color='blue', line_width=5)
box = pv.Cube(bounds=(0, 5, 0, 5, 10, 11))
plotter.add_mesh(box, color='white', style='wireframe', line_width=2)

# Volume render the Gaussian field
plotter.add_volume(grid, scalars='pdf', cmap='viridis', opacity='sigmoid')
plotter.show()
