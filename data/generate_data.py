import numpy as np 
from PIL import Image  
import json 
import os 

if __name__ == "__main__":
    folder_prefix = './lego'
    
    os.mkdir('dozer')
    os.mkdir('dozer/images')
    os.mkdir('dozer/images2')
    os.mkdir('dozer/images4')
    os.mkdir('dozer/images8')
    
    # images = data["images"]
    # poses = data["poses"]
    # focal = data["focal"]
    # # envs = data["env"]

    focal = 800/(2*np.tan(0.6911112070083618/2))

    output_dict = {
        "w": 800,
        "h": 800,
        "fl_x": focal,
        "fl_y": focal,
        "cx": 400,
        "cy": 400,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "applied_transform": [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0]
        ],
        "frames":[]
    }

    with open(os.path.join(folder_prefix, 'transforms_train.json')) as f:
        data_orig = json.load(f)
    
    for i in range(len(data_orig['frames'])):
        transform_matrix = data_orig['frames'][i]['transform_matrix']
        image_pth = os.path.join(folder_prefix,data_orig['frames'][i]['file_path'])
        image = Image.open(f"{image_pth}.png")
        image_2 = image.resize((400,400))
        image_4 = image.resize((200,200))
        image_8 = image.resize((100,100))

        output_image_fn = f"frame_{i+1:05d}.png"

        frame = {
            "file_path": f"images/frame_{i+1:05d}.png",
            "original_fn": image_pth,
            "transform_matrix": transform_matrix,
            "colmap_im_id": i+1 
        }

        output_dict['frames'].append(frame)

        image.save(os.path.join("dozer/images", output_image_fn))
        image_2.save(os.path.join("dozer/images2", output_image_fn))
        image_4.save(os.path.join("dozer/images4", output_image_fn))
        image_8.save(os.path.join("dozer/images8", output_image_fn))
    with open('dozer/transform.json', 'w+') as f:
        json.dump(output_dict, f)