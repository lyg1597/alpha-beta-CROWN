from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from simple_model2_alphatest2 import AlphaModel, DepthModel
from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
from scipy.spatial.transform import Rotation 
from collections import defaultdict
from image_helpper import get_rect_set, \
    get_viewmat, \
    get_bound_depth_step, \
    computeT_new, \
    computeT_new_optimized, \
    computeT, \
    get_elem_before_linear
import time 

dt = {
    "transform": [
        [
            0.9990379810333252,
            -0.030679119750857353,
            -0.031336162239313126,
            -79.60553741455078
        ],
        [
            -0.030679119750857353,
            0.021658122539520264,
            -0.999294638633728,
            1.5170279741287231
        ],
        [
            0.031336162239313126,
            0.999294638633728,
            0.020696043968200684,
            -28.584909439086914
        ]
    ],
    "scale": 0.007225637783593824
}

def compute_tile_color(
    # Input info
    cam_inp,
    eps,

    # Gaussian input 
    means_strip,
    opacities_strip,
    scales_strip,
    quats_strip,
    colors_strip,

    # Add camera info 
    f,
    w,
    h,
    width,
    height,

    # Add parameters
    pix_coord,
    tile_size,
    gauss_step,
):
    tile_coord = pix_coord[h:h+tile_size, w:w+tile_size].flatten(0,-2)
    N = means_strip.shape[0]

    overall_alpha_lb = torch.zeros((1,tile_size*tile_size, 0, 1)).to(means_strip.device)
    overall_alpha_ub = torch.zeros((1,tile_size*tile_size, 0, 1)).to(means_strip.device)

    overall_depth_lA = torch.zeros((1,0,6)).to(means_strip.device)
    overall_depth_uA = torch.zeros((1,0,6)).to(means_strip.device)
    overall_depth_lbias = torch.zeros((1,0)).to(means_strip.device)
    overall_depth_ubias = torch.zeros((1,0)).to(means_strip.device)
    
    for j in range(0, N, gauss_step):
        data_pack = {
            'opacities': torch.Tensor(opacities_strip[j:j+gauss_step]),
            'means': torch.Tensor(means_strip[j:j+gauss_step]),
            'scales':torch.Tensor(scales_strip[j:j+gauss_step]),
            'quats':torch.Tensor(quats_strip[j:j+gauss_step]),
        } 

        model_alpha = AlphaModel(
            data_pack = data_pack,
            fx = f,
            fy = f,
            width = width,
            height = height,
            tile_coord = tile_coord 
        )

        model_depth = DepthModel(model_alpha)

        inp_alpha = torch.clone(cam_inp)
        # print(">>>>>> Starting Bounded Module")
        tmp = time.time()
        model_alpha_bounded = BoundedModule(model_alpha, inp_alpha, device=means_strip.device)
        print(f"time for create bounded model {time.time()-tmp}")
        # print(">>>>>> Starting PerturbationLpNorm")
        ptb_alpha = PerturbationLpNorm(norm=np.inf, x_L=inp_alpha-eps, x_U=inp_alpha+eps)
        # print(">>>>>> Starting BoundedTensor")
        inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
        prediction = model_alpha_bounded(inp_alpha)
        tmp = time.time()
        lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(inp_alpha, ), method='crown')
        print(f'time for compute bound {time.time()-tmp}')
        # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
        overall_alpha_lb = torch.cat((overall_alpha_lb, lb_alpha), dim=2)
        overall_alpha_ub = torch.cat((overall_alpha_ub, ub_alpha), dim=2)
        overall_alpha_lb = overall_alpha_lb.clip(min=0.0, max=0.99)
        overall_alpha_ub = overall_alpha_ub.clip(min=0.0, max=0.99)

        inp_depth = torch.clone(cam_inp)
        # print(">>>>>> Starting Bounded Module")
        model_depth_bounded = BoundedModule(model_depth, inp_depth, device=means_strip.device)
        # print(">>>>>> Starting PerturbationLpNorm")
        ptb_depth = PerturbationLpNorm(norm=np.inf, x_L=inp_depth-eps, x_U=inp_depth+eps)
        # print(">>>>>> Starting BoundedTensor")
        inp_depth = BoundedTensor(inp_depth, ptb_depth)
        prediction = model_depth_bounded(inp_depth)
        required_A = defaultdict(set)
        required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
        lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, ), method='crown', return_A=True, needed_A_dict=required_A)

        depth_lA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lA']
        depth_uA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['uA']
        depth_lbias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lbias']
        depth_ubias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['ubias']

        overall_depth_lA = torch.cat((overall_depth_lA, depth_lA), dim=1)
        overall_depth_uA = torch.cat((overall_depth_uA, depth_uA), dim=1)
        overall_depth_lbias = torch.cat((overall_depth_lbias, depth_lbias), dim=1)
        overall_depth_ubias = torch.cat((overall_depth_ubias, depth_ubias), dim=1)

    bounds_alpha = torch.cat((overall_alpha_lb, overall_alpha_ub), dim=0)
    # concrete_before, possible_before = get_elem_before_linear(
    #     ptb_depth, 
    #     overall_depth_lA, 
    #     overall_depth_uA, 
    #     overall_depth_lbias, 
    #     overall_depth_ubias, 
    # )
    # res_T = computeT(concrete_before, possible_before, bounds_alpha)
    step_L, step_U = get_bound_depth_step(
        ptb_depth,
        overall_depth_lA,
        overall_depth_uA,
        overall_depth_lbias,
        overall_depth_ubias,
    )
    # print(f"time for compute step bound {time.time()-tmp}")
    # tmp = time.time()
    # res_Tl = computeT_new(overall_alpha_ub, step_U)
    tmp_res_Tl = computeT_new_optimized(overall_alpha_ub, step_U)
    tmp_res_Tu = computeT_new_optimized(overall_alpha_lb, step_L)
    # res_Tu = computeT_new(overall_alpha_lb, step_L)
    # print(f"time for compute T bound{time.time()-tmp}")
    # res_TL = (torch.ones((1,1,1,1)).to(overall_alpha_ub.device)-overall_alpha_ub*step_U[:,None].to(overall_alpha_ub.device)).prod(dim=3)
    res_T = torch.cat((tmp_res_Tl, tmp_res_Tu), dim=0)
    # res_T = computeT(concrete_before, possible_before, bounds_alpha)
    bounds_colors = torch.stack((colors_strip, colors_strip), dim=0)
    bounds_colors = bounds_colors[:,None]
    tile_color = (res_T*bounds_alpha*bounds_colors).sum(dim=2)
    tile_color_lb = tile_color[0]
    tile_color_ub = tile_color[1]
    tile_color_lb = tile_color_lb.reshape(tile_size, tile_size, -1)[:,:,:3]
    tile_color_ub = tile_color_ub.reshape(tile_size, tile_size, -1)[:,:,:3]
    tile_color_lb = tile_color_lb.detach().cpu().numpy()
    tile_color_ub = tile_color_ub.detach().cpu().numpy()        
    return tile_color_lb, tile_color_ub

if __name__ == "__main__":
    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '/home/younger/work/nerfstudio/outputs/triangular_data4/splatfacto/2025-01-19_232156')
    checkpoint = "step-000029999.ckpt"
    
    checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

    # Filter unnecessary gaussians 
    # color_mask = torch.norm(colors, dim=1)>0.1
    # means_trans = torch.inverse(torch.tensor(transform_ap).to(means.device))@means.transpose(0,1)/scale

    # means = means[color_mask,:]
    # quats = quats[color_mask,:]
    # opacities = opacities[color_mask,:]
    # scales = scales[color_mask,:]
    # colors = colors[color_mask,:]


    camera_pose = np.array([
        [
            0.0,
            0.0,
            1.0,
            200.0
        ],
        [
            0.0,
            1.0,
            0.0,
           5.0
        ],
        [
            -1.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])

    # transform = np.array(dt['transform'])
    # scale = dt['scale']
    camera_pose_transformed = transform@camera_pose
    camera_pose_transformed[:3,3] *= scale 
    camera_pose_transformed = torch.Tensor(camera_pose_transformed)[None].to(means.device)

    width=96
    height=96
    f = 120

    eps = 0.00001
    tile_size = 8
    gauss_step = 400

    # camera_to_worlds = torch.Tensor(camera_pose)[None].to(means.device)
    camera_to_world = [
        [0.09323342123480016,
        -0.21989920695796833,
        0.9710570878913922,
        0.3464003611615004,],
        [0.9956442784272173,
        0.02059164687201054,
        -0.09093104483202696,
        -0.0148054846552511,],
        [-2.7755575615628914e-16,
        0.975305245991405,
        0.22086121692512006,
        -0.13118252322656915,],
        [0.0,
        0.0,
        0.0,
        1.0]
    ]
    camera_to_world = torch.Tensor(camera_pose_transformed)[None].to(means.device)

    view_mats = get_viewmat(camera_pose_transformed)
    Ks = torch.tensor([[
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]
    ]]).to(torch.device('cuda'))

    camera_pos = view_mats[0,:3,3].detach().cpu().numpy()
    camera_ori = Rotation.from_matrix(view_mats[0,:3,:3].detach().cpu().numpy()).as_euler('xyz')
    cam_inp = [
        camera_ori[0], 
        camera_ori[1], 
        camera_ori[2], 
        camera_pos[0], 
        camera_pos[1], 
        camera_pos[2]
    ]
    cam_inp = torch.Tensor(cam_inp)[None].to('cuda')


    with torch.no_grad():
        res = rasterize_gaussians_pytorch_rgb(
            means, 
            quats/ quats.norm(dim=-1, keepdim=True),
            torch.exp(scales),
            torch.sigmoid(opacities).squeeze(-1),
            colors,
            view_mats, 
            Ks,
            width,
            height
        )
    res_rgb = res['render']
    print(res_rgb.shape)
    res_rgb = res_rgb[:,...,:3]
    res_rgb = res_rgb.detach().cpu().numpy()
    plt.figure(0)
    plt.imshow(res_rgb)
    plt.show()

    # Get all the pix_coord 
    pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to(means.device)
    # Get the rectangles of gaussians under uncertainty 
    rect, mask = get_rect_set(
        cam_inp-eps,
        cam_inp+eps,
        means,
        scales,
        quats,
        f,
        f,
        width,
        height 
    )
    render_color_lb = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    render_color_ub = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    means = means[mask]
    quats = quats[mask]
    opacities = opacities[mask]
    scales = scales[mask]
    colors = colors[mask]
    for h in range(0, height, tile_size):
        for w in range(0, width, tile_size):
            # if h<20:
            #     continue
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+tile_size-1), rect[1][..., 1].clip(max=h+tile_size-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
            if not in_mask.sum() > 0:
                continue
            means_strip = means[in_mask]
            quats_strip = quats[in_mask]
            opacities_strip = opacities[in_mask]
            scales_strip = scales[in_mask]
            colors_strip = colors[in_mask]
            print(f">>>>>>>> {h}, {w}, {means_strip.shape[0]}")
            # means_strip = means
            # quats_strip = quats
            # opacities_strip = opacities
            # scales_strip = scales
            # colors_strip = colors

            tile_color_lb, tile_color_ub = compute_tile_color(
                cam_inp,
                eps,
                means_strip,
                opacities_strip,
                scales_strip,
                quats_strip,
                colors_strip,
                f,
                w,
                h,
                width,
                height,
                pix_coord,
                tile_size,
                gauss_step
            )
            # tile_coord = pix_coord[h:h+tile_size, w:w+tile_size].flatten(0,-2)
            # N = means_strip.shape[0]

            # overall_alpha_lb = torch.zeros((1,tile_size*tile_size, 0, 1)).to(means.device)
            # overall_alpha_ub = torch.zeros((1,tile_size*tile_size, 0, 1)).to(means.device)

            # overall_depth_lA = torch.zeros((1,0,6)).to(means.device)
            # overall_depth_uA = torch.zeros((1,0,6)).to(means.device)
            # overall_depth_lbias = torch.zeros((1,0)).to(means.device)
            # overall_depth_ubias = torch.zeros((1,0)).to(means.device)
            
            # for j in range(0, N, gauss_step):
            #     data_pack = {
            #         'opacities': torch.Tensor(opacities_strip[j:j+gauss_step]),
            #         'means': torch.Tensor(means_strip[j:j+gauss_step]),
            #         'scales':torch.Tensor(scales_strip[j:j+gauss_step]),
            #         'quats':torch.Tensor(quats_strip[j:j+gauss_step]),
            #     } 

            #     model_alpha = AlphaModel(
            #         data_pack = data_pack,
            #         fx = f,
            #         fy = f,
            #         width = width,
            #         height = height,
            #         tile_coord = tile_coord 
            #     )

            #     model_depth = DepthModel(model_alpha)

            #     inp_alpha = torch.clone(cam_inp)
            #     # print(">>>>>> Starting Bounded Module")
            #     model_alpha_bounded = BoundedModule(model_alpha, inp_alpha, device=means.device)
            #     # print(">>>>>> Starting PerturbationLpNorm")
            #     ptb_alpha = PerturbationLpNorm(norm=np.inf, x_L=inp_alpha-eps, x_U=inp_alpha+eps)
            #     # print(">>>>>> Starting BoundedTensor")
            #     inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
            #     prediction = model_alpha_bounded(inp_alpha)
            #     lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(inp_alpha, ), method='ibp')
            #     # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
            #     overall_alpha_lb = torch.cat((overall_alpha_lb, lb_alpha), dim=2)
            #     overall_alpha_ub = torch.cat((overall_alpha_ub, ub_alpha), dim=2)

            #     inp_depth = torch.clone(cam_inp)
            #     # print(">>>>>> Starting Bounded Module")
            #     model_depth_bounded = BoundedModule(model_depth, inp_depth, device=colors.device)
            #     # print(">>>>>> Starting PerturbationLpNorm")
            #     ptb_depth = PerturbationLpNorm(norm=np.inf, x_L=inp_depth-eps, x_U=inp_depth+eps)
            #     # print(">>>>>> Starting BoundedTensor")
            #     inp_depth = BoundedTensor(inp_depth, ptb_depth)
            #     prediction = model_depth_bounded(inp_depth)
            #     required_A = defaultdict(set)
            #     required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
            #     lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, ), method='crown', return_A=True, needed_A_dict=required_A)

            #     depth_lA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lA']
            #     depth_uA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['uA']
            #     depth_lbias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lbias']
            #     depth_ubias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['ubias']

            #     overall_depth_lA = torch.cat((overall_depth_lA, depth_lA), dim=1)
            #     overall_depth_uA = torch.cat((overall_depth_uA, depth_uA), dim=1)
            #     overall_depth_lbias = torch.cat((overall_depth_lbias, depth_lbias), dim=1)
            #     overall_depth_ubias = torch.cat((overall_depth_ubias, depth_ubias), dim=1)

            # bounds_alpha = torch.cat((overall_alpha_lb, overall_alpha_ub), dim=0)
            # # concrete_before, possible_before = get_elem_before_linear(
            # #     ptb_depth, 
            # #     overall_depth_lA, 
            # #     overall_depth_uA, 
            # #     overall_depth_lbias, 
            # #     overall_depth_ubias, 
            # # )
            # step_L, step_U = get_bound_depth_step(
            #     ptb_depth,
            #     overall_depth_lA,
            #     overall_depth_uA,
            #     overall_depth_lbias,
            #     overall_depth_ubias,
            # )
            # res_Tl = computeT_new(overall_alpha_ub, step_U)
            # res_Tu = computeT_new(overall_alpha_lb, step_L)
            # # res_TL = (torch.ones((1,1,1,1)).to(overall_alpha_ub.device)-overall_alpha_ub*step_U[:,None].to(overall_alpha_ub.device)).prod(dim=3)
            # res_T = torch.cat((res_Tl, res_Tu), dim=0)
            # # res_T = computeT(concrete_before, possible_before, bounds_alpha)
            # bounds_colors = torch.stack((colors_strip, colors_strip), dim=0)
            # bounds_colors = bounds_colors[:,None]
            # tile_color = (res_T*bounds_alpha*bounds_colors).sum(dim=2)
            # tile_color_lb = tile_color[0]
            # tile_color_ub = tile_color[1]
            # # depth_order = torch.argsort(overall_depth, dim=1).squeeze()
            # # sorted_alpha = overall_alpha[0,:,depth_order,:]
            # # sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
            # # sorted_color = colors_strip[depth_order,:]
            # # rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
            # tile_color_lb = tile_color_lb.reshape(tile_size, tile_size, -1)[:,:,:3]
            # tile_color_ub = tile_color_ub.reshape(tile_size, tile_size, -1)[:,:,:3]
            render_color_lb[h:h+tile_size, w:w+tile_size] = tile_color_lb
            render_color_ub[h:h+tile_size, w:w+tile_size] = tile_color_ub
            plt.imshow(render_color_lb)
            plt.savefig('res_lb.png')
            plt.imshow(render_color_ub)
            plt.savefig('res_ub.png')
    plt.figure(0)
    plt.imshow(render_color_lb)
    plt.figure(1)
    plt.imshow(render_color_ub)
    plt.show()
                # inp_alpha = torch.clone(cam_inp)
                # print(">>>>>> Starting Bounded Module")
                # model_alpha_bounded = BoundedModule(model_alpha, inp_alpha, device=means.device)
                # print(">>>>>> Starting PerturbationLpNorm")
                # ptb_alpha = PerturbationLpNorm(norm=np.inf, x_L=inp_alpha-eps, x_U=inp_alpha+eps)
                # print(">>>>>> Starting BoundedTensor")
                # inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
                # prediction = model_alpha_bounded(inp_alpha)
                # lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(inp_alpha, ), method='ibp')
                # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)

                # inp_depth = torch.clone(cam_inp)
                # print(">>>>>> Starting Bounded Module")
                # model_depth_bounded = BoundedModule(model_depth, inp_depth, device=colors.device)
                # print(">>>>>> Starting PerturbationLpNorm")
                # ptb_depth = PerturbationLpNorm(norm=np.inf, x_L=inp_depth-eps, x_U=inp_depth+eps)
                # print(">>>>>> Starting BoundedTensor")
                # inp_depth = BoundedTensor(inp_depth, ptb_depth)
                # prediction = model_depth_bounded(inp_depth)
                # required_A = defaultdict(set)
                # required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
                # lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, ), method='crown', return_A=True, needed_A_dict=required_A)
 
            # concrete_before, possible_before = get_elem_before_linear(ptb_depth, A_depth, model_depth_bounded)
            # print(concrete_before, possible_before)
            # res_T = computeT(concrete_before, possible_before, bounds_alpha)
            
            # res_2d = colors
            # bounds_res_2d = torch.stack((res_2d, res_2d), dim=0)
            # bounds_res_2d = bounds_res_2d[:,None]
            # tile_color = (res_T*bounds_alpha*bounds_res_2d).sum(dim=2)

    # overall_mask = res['overall_mask']
    # means = means[overall_mask].detach()
    # colors = colors[overall_mask].detach()
    # quats = quats[overall_mask].detach()
    # opacities = opacities[overall_mask].detach()
    # scales = scales[overall_mask].detach()

    # means = means[:5000]
    # colors = colors[:5000]
    # quats = quats[:5000]
    # opacities = opacities[:5000]
    # scales = scales[:5000]

    # data_pack = {
    #     'opacities': torch.Tensor(opacities),
    #     'means': torch.Tensor(means),
    #     'scales':torch.Tensor(scales),
    #     'quats':torch.Tensor(quats),
    #     # 'tile_coords':torch.Tensor(tile_coords)
    # }

    # model_alpha = AlphaModel(
    #     data_pack = data_pack,
    #     fx = f,
    #     fy = f,
    #     width = width,
    #     height = height
    # )

    # transform = np.array(dt['transform'])
    # scale = dt['scale']
    # camera_pose_transformed = transform@camera_pose
    # camera_pose_transformed[:3,3] *= scale 
    # camera_pose_transformed = torch.Tensor(camera_pose_transformed)
    # camera_pose_transformed = get_viewmat(camera_pose_transformed[None])
    
    # camera_pos = camera_pose_transformed[0,:3,3].detach().cpu().numpy()
    # camera_ori = Rotation.from_matrix(camera_pose_transformed[0,:3,:3]).as_euler('xyz')
    # camera_pose_transformed = [
    #     camera_ori[0], 
    #     camera_ori[1], 
    #     camera_ori[2], 
    #     camera_pos[0], 
    #     camera_pos[1], 
    #     camera_pos[2]
    # ]
    # camera_pose_transformed = torch.Tensor(camera_pose_transformed)[None].to('cuda')

    # model_depth = DepthModel(model_alpha)
    
    # res_alpha = model_alpha(camera_pose_transformed)
    # print("###### Alpha")
    # res_depth = model_depth(camera_pose_transformed)
    # print("###### Depth")
    # depth_order = torch.argsort(res_depth, dim=1).squeeze()
    # sorted_alpha = res_alpha[0,:,depth_order,:]
    # sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    # sorted_color = colors[depth_order,:]
    # rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
    # rgb_color = rgb_color.reshape(width, height, -1)[:,:,:3]
    # rgb_color = rgb_color.detach().cpu().numpy()
    # plt.figure(0)
    # plt.imshow(rgb_color)
    # plt.show()

    # # torch.onnx.export(model_alpha, view_mats, 'model.onnx') 

    # # my_input = torch.clone(res_2d[model.overall_mask])
    # inp_alpha = torch.clone(camera_pose_transformed)
    # print(">>>>>> Starting Bounded Module")
    # model_alpha_bounded = BoundedModule(model_alpha, inp_alpha, device=means.device)
    # print(">>>>>> Starting PerturbationLpNorm")
    # ptb_alpha = PerturbationLpNorm(norm=np.inf, eps=0.00001)
    # print(">>>>>> Starting BoundedTensor")
    # inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
    # prediction = model_alpha_bounded(inp_alpha)
    # lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(inp_alpha, ), method='ibp')
    # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
    # # bounds_alpha = 
    # # lb2, ub2 = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # # print(len(torch.where((lb<lb2)|(ub>ub2))[0]))
    # # print(torch.max(lb2-lb), torch.max(ub-ub2))
    
    # print(">>>>>> Done")
    # print(lb_alpha.shape)
    # print(ub_alpha.shape)

    # # model_depth = DepthModel(model_alpha)
    # # view_mats = model_alpha.viewmat
    # # with torch.no_grad():
    # #     res = model_depth(view_mats)
    # # print(res.shape)
    # inp_depth = torch.clone(camera_pose_transformed)
    # print(">>>>>> Starting Bounded Module")
    # model_depth_bounded = BoundedModule(model_depth, inp_depth, device=colors.device)
    # print(">>>>>> Starting PerturbationLpNorm")
    # ptb_depth = PerturbationLpNorm(norm=np.inf, eps=0.00001)
    # print(">>>>>> Starting BoundedTensor")
    # inp_depth = BoundedTensor(inp_depth, ptb_depth)
    # prediction = model_depth_bounded(inp_depth)
    # required_A = defaultdict(set)
    # required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
    # lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, ), method='crown', return_A=True, needed_A_dict=required_A)
    
    # lb_depth = lb_depth.detach().cpu().numpy()    
    # ub_depth = ub_depth.detach().cpu().numpy()    
    # bounds_depth = np.vstack((lb_depth, ub_depth)).T
    # bounds_depth = bounds_depth.tolist()
    # bounds_depth = [elem+[i] for i, elem in enumerate(bounds_depth)]
    # # sorted_bounds = sort_bounds(bounds_depth)

    # # concrete_before, possible_before = get_elem_before(bounds_depth)
    # concrete_before, possible_before = get_elem_before_linear(ptb_depth, A_depth, model_depth_bounded)
    # print(concrete_before, possible_before)
    # res_T = computeT(concrete_before, possible_before, bounds_alpha)
    
    # res_2d = colors
    # bounds_res_2d = torch.stack((res_2d, res_2d), dim=0)
    # bounds_res_2d = bounds_res_2d[:,None]
    # tile_color = (res_T*bounds_alpha*bounds_res_2d).sum(dim=2)

    # tile_color_lb = tile_color[0,:,:3].reshape((width,height,-1))
    # tile_color_lb = tile_color_lb.detach().cpu().numpy()
    # tile_color_ub = tile_color[1,:,:3].reshape((width,height,-1))
    # tile_color_ub = tile_color_ub.detach().cpu().numpy()

    # empirical_lb = np.zeros(tile_color_lb.shape)+1e10
    # empirical_ub = np.zeros(tile_color_lb.shape)
    # empirical_alpha_lb = np.zeros(lb_alpha.shape)+1e10
    # empirical_alpha_ub = np.zeros(ub_alpha.shape)
    # lb_alpha = lb_alpha.detach().cpu().numpy()
    # ub_alpha = ub_alpha.detach().cpu().numpy()
    # for i in range(1000):
    #     tmp_input = my_input.repeat(1,1)
    #     delta = torch.zeros((1,6))
    #     # delta[:,:3,3] = torch.rand((1,3))*eps*2-eps
    #     delta[:,:3] = torch.rand((1,3))*0.0*2-0.0
    #     delta[:,3:] = torch.rand((1,3))*1.0*2-1.0
    #     delta = delta.to(model_depth.device)
    #     tmp_input = tmp_input+delta 
    #     res_alpha = model_alpha(tmp_input)
    #     res_depth = model_depth(tmp_input)
    #     depth_order = torch.argsort(res_depth, dim=1).squeeze()
    #     sorted_alpha = res_alpha[0,:,depth_order,:]
    #     sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    #     sorted_color = colors[depth_order,:]
    #     alphac = res_alpha[0]*colors[None]
    #     sorted_alphac = alphac[:,depth_order]
    #     rgb_color = (sorted_T * sorted_alphac).sum(dim=1)
    #     res_alpha = res_alpha.detach().cpu().numpy()
    #     empirical_alpha_lb = np.minimum(empirical_alpha_lb, res_alpha)
    #     empirical_alpha_ub = np.maximum(empirical_alpha_ub, res_alpha)
    #     rgb_color = rgb_color.reshape(width, height, -1)[:,:,:3]
    #     rgb_color = rgb_color.detach().cpu().numpy()
    #     empirical_lb = np.minimum(empirical_lb, rgb_color)
    #     empirical_ub = np.maximum(empirical_ub, rgb_color)
    #     valid_bound = np.all(rgb_color>=tile_color_lb) and np.all(rgb_color<=tile_color_ub)
    #     if not valid_bound:
    #         print("Bound Violated")
    #         break

    # diff_compemp_ub = (ub_alpha-empirical_alpha_ub).reshape(width,height,-1)
    # diff_compemp_lb = (empirical_alpha_lb-lb_alpha).reshape(width,height,-1)

    # tile_color_ub[:,:,1:] = 0

    # plt.figure(1)
    # plt.imshow(tile_color_lb)
    # plt.title("computed lb alpha-crown")
    # plt.figure(2)
    # plt.imshow(tile_color_ub)
    # plt.title("computed ub alpha-crown handle 0")
    # plt.figure(3)
    # plt.imshow(empirical_lb)
    # plt.title("empirical lb")
    # plt.figure(4)
    # plt.imshow(empirical_ub)
    # plt.title("empirical ub")
    # plt.figure(5)
    # plt.imshow(diff_compemp_lb)
    # plt.title('diff_comp_emp_lb')
    # plt.figure(6)
    # plt.imshow(diff_compemp_ub)
    # plt.title('diff_comp_emp_ub')
    # plt.show()
