import numpy as np
from interval import interval

# Given values
fx, fy = 40, 40
x0, y0, z0 = -2, 0, 10

# Define intervals for dx, dy, dz
dx = interval[-1.0, 1.0]
dy = interval[-1.0, 1.0]
dz = interval[-1.0, 1.0]

# Matrix S (same as before)
S = np.array([[0.4493, 0.0000, 0.0000],
            [0.0000, 0.6703, 0.0000],
            [0.0000, 0.0000, 0.4493]])

# Define J0 (numeric matrix)
J0 = np.array([[fx * z0, 0, -fx * x0],
               [0, fy * z0, -fy * y0]])

# dJ with interval arithmetic (use list of lists to hold intervals)
dJ = [[fx * dz, 0, -fx * dx],
      [0, fy * dz, -fy * dy]]
dJT= [[fx * dz, 0],
      [0, fy * dz],
      [-fx * dx, -fy * dy]]

# Compute A0 = J0 * S * J0.T (regular matrix operations)
A0 = J0 @ S @ J0.T
A0_inv=np.linalg.inv(A0)
print('A0_inv:',A0_inv)

# Compute D manually, accounting for intervals in dJ
D = [[interval[0, 0], interval[0, 0]],  # Initialize empty D matrix with intervals
     [interval[0, 0], interval[0, 0]]]

dJS=[[interval[0, 0], interval[0, 0], interval[0, 0]],  
      [interval[0, 0], interval[0, 0], interval[0, 0]]]
for k in range(2):
      for l in range(3):
            for m in range(3):
                  dJS[k][l]+=dJ[k][m]*float(S[m,l])

for i in range(2):  # 2 rows for D
      for j in range(2):  # 2 columns for D
            for k in range(3):
                  D[i][j]+=float((J0 @ S)[i,k]) * dJT[k][j]+ \
                        dJ[i][k]*float((S@J0.T)[k,j])+\
                        dJS[i][k]*dJT[k][j]
print('D:',D)

res1 = [[interval[0, 0], interval[0, 0]],
        [interval[0, 0], interval[0, 0]]]
res2 = [[interval[0, 0], interval[0, 0]],
        [interval[0, 0], interval[0, 0]]]
tmp2=[[interval[0, 0], interval[0, 0]],  
      [interval[0, 0], interval[0, 0]]]
tmp3=[[interval[0, 0], interval[0, 0]],  
      [interval[0, 0], interval[0, 0]]]
tmp4=[[interval[0, 0], interval[0, 0]],  
      [interval[0, 0], interval[0, 0]]]
tmp5=[[interval[0, 0], interval[0, 0]],  
      [interval[0, 0], interval[0, 0]]]
for k in range(2):
      for l in range(2):
            for m in range(2):
                  tmp2[k][l]+=float(A0_inv[k, m])*D[m][l]
for k in range(2):
      for l in range(2):
            for m in range(2):
                  tmp3[k][l]+=tmp2[k][m]*float(A0_inv[m, l])
for k in range(2):
      for l in range(2):
            for m in range(2):
                  tmp4[k][l]+=tmp3[k][m]*D[m][l]
for k in range(2):
      for l in range(2):
            for m in range(2):
                  tmp5[k][l]+=tmp4[k][m]*float(A0_inv[m,l])



for i in range(2):
      for j in range(2):
            res1[i][j] = float(A0_inv[i, j])*interval[1,1]
            res1[i][j]+= -tmp3[i][j]

for i in range(2):
      for j in range(2):
            res2[i][j] = float(A0_inv[i, j])*interval[1,1]
            res2[i][j]+= -tmp3[i][j]+tmp5[i][j]

print('res1:',res1)
print('res2:',res2)