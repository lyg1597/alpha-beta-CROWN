from PIL import Image
import numpy as np 
import os 
script_dir = os.path.dirname(os.path.realpath(__file__))

img_emp_lb = Image.open(os.path.join(script_dir, './dozer1_fine_emp_lb.png'))
img_emp_ub = Image.open(os.path.join(script_dir, './dozer1_fine_emp_ub.png'))

img_lb = Image.open(os.path.join(script_dir, './dozer1_fine_lb.png'))
img_ub = Image.open(os.path.join(script_dir, './dozer1_fine_ub.png'))

img_emp_lb_array = np.array(img_emp_lb)
img_emp_ub_array = np.array(img_emp_ub)

img_lb_array = np.array(img_lb)
img_ub_array = np.array(img_ub)

img_lb_fixed_array = np.minimum(img_emp_lb_array, img_lb_array)
img_ub_fixed_array = np.maximum(img_emp_ub_array, img_ub_array)

img_lb_fixed = Image.fromarray(img_lb_fixed_array)
img_ub_fixed = Image.fromarray(img_ub_fixed_array)

img_lb_fixed.save(os.path.join(script_dir,'./dozer_fine_lb_fixed2.png'))
img_ub_fixed.save(os.path.join(script_dir,'./dozer_fine_ub_fixed2.png'))
