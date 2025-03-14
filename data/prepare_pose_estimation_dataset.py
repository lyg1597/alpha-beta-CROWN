import numpy as np 
from PIL import Image  
import json 
import os
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    folder_prefix = './lego'

    os.mkdir('lego_dataset')
    os.mkdir('lego_dataset/images')
    os.mkdir('lego_dataset/poses')

    with open(os.path.join(folder_prefix, 'transforms_train.json')) as f:
        data_orig = json.load(f)

    idx = 1
    for i in range(len(data_orig['frames'])):
        transform_matrix = data_orig['frames'][i]['transform_matrix']
        image_pth = os.path.join(folder_prefix,data_orig['frames'][i]['file_path'])
        image = Image.open(f"{image_pth}.png")
        image_16 = image.resize((50,50))

        mat = np.array(transform_matrix)
        x,y,z = mat[0,3],mat[1,3],mat[2,3]
        roll, pitch, yaw = Rotation.from_matrix(mat[:3,:3]).as_euler('xyz')
        
        image_16.save(os.path.join('lego_dataset/images', f"image_{idx}.png"))
        with open(os.path.join('lego_dataset/poses', f"image_{idx}.txt"), 'w+') as f:
            f.write(f"{x} {y} {z} {roll} {pitch} {yaw}\n")

        idx += 1

    with open(os.path.join(folder_prefix, 'transforms_val.json')) as f:
        data_orig = json.load(f)

    idx = 1
    for i in range(len(data_orig['frames'])):
        transform_matrix = data_orig['frames'][i]['transform_matrix']
        image_pth = os.path.join(folder_prefix,data_orig['frames'][i]['file_path'])
        image = Image.open(f"{image_pth}.png")
        image_16 = image.resize((50,50))

        mat = np.array(transform_matrix)
        x,y,z = mat[0,3],mat[1,3],mat[2,3]
        roll, pitch, yaw = Rotation.from_matrix(mat[:3,:3]).as_euler('xyz')
        
        image_16.save(os.path.join('lego_dataset/images', f"image_{idx}.png"))
        with open(os.path.join('lego_dataset/poses', f"image_{idx}.txt"), 'w+') as f:
            f.write(f"{x} {y} {z} {roll} {pitch} {yaw}\n")

        idx += 1
        # output_image_fn = f"frame_{i+1:05d}.png"

        # frame = {
        #     "file_path": f"images/frame_{i+1:05d}.png",
        #     "original_fn": image_pth,
        #     "transform_matrix": transform_matrix,
        #     "colmap_im_id": i+1 
        # }

        # output_dict['frames'].append(frame)

        # image.save(os.path.join("dozer/images", output_image_fn))
        # image_2.save(os.path.join("dozer/images2", output_image_fn))
        # image_4.save(os.path.join("dozer/images4", output_image_fn))
        # image_8.save(os.path.join("dozer/images8", output_image_fn))