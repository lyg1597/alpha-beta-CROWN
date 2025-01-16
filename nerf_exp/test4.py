import numpy as np
import pyvista as pv

# 1) Define the domain
x = np.linspace(-3, -1, 50)
y = np.linspace(-1, 1, 50)

# 2) Create a 2D meshgrid (X, Y), then compute Z = X * Y
X, Y = np.meshgrid(x, y)
Z = X * Y  # The function you want to plot

# 3) Convert these arrays into a PyVista StructuredGrid
#    Note: (X, Y, Z) should each be 2D arrays of the same shape
grid = pv.StructuredGrid(X, Y, Z)

# 4) Plot the surface
p = pv.Plotter()
p.add_mesh(grid, show_edges=True, cmap="viridis")
# Add axes lines
# --- 3) Create axes lines (X, Y, Z) starting at the origin ---
x_axis_line = pv.Line(pointa=(0, 0, 0), pointb=(5, 0, 0))
y_axis_line = pv.Line(pointa=(0, 0, 0), pointb=(0, 5, 0))
z_axis_line = pv.Line(pointa=(0, 0, 0), pointb=(0, 0, 5))
p.add_mesh(x_axis_line, color='red', line_width=5, name='x-axis')
p.add_mesh(y_axis_line, color='green', line_width=5, name='y-axis')
p.add_mesh(z_axis_line, color='blue', line_width=5, name='z-axis')
p.show()