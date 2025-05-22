import os
import wget
import sys
sys.path.append("/large/naru/HOGraspNet")
from config import cfg
import requests
from requests.exceptions import ConnectionError
from tqdm.autonotebook import tqdm
import time
import math


def check_args(arg_type, arg_subject):
    try:  
        if arg_type == 0:            
            target_url_set = ["images_augmented", "annotations", "extra"]
        elif arg_type == 1:
            target_url_set = cfg.base_url_set
        elif arg_type == 2:            
            target_url_set = ["images_augmented"]
        elif arg_type == 3:            
            target_url_set = ["annotations"]
        elif arg_type == 4:            
            target_url_set = ["extra"]
        elif arg_type == 5:            
            target_url_set = ["images"]
        else:
            raise Exception("wrong type argument")
    except Exception as e:
        print("ERROR: wrong --type argument format. Please check the --help")

    try:  
        if arg_subject == "all":            
            subjects = cfg.subject_types
        else:
            if arg_subject == "small":
                subjects = [43, 63, 83, 84, 93]
            elif "-" in arg_subject:
                subjects = arg_subject.split("-")
                subjects = list(range(int(subjects[0]), int(subjects[1])+1))
            else:
                subjects = arg_subject.split(",")
                subjects = list(map(int, subjects))
    except Exception as e:
        print("ERROR: wrong --subject argument format. Please check the --help")

    return target_url_set, subjects


def download_urls(urls, output_folder, max_tries=7):
    """
    Download file from a URL to file_name,
    source refer from https://github.com/facebookresearch/ContactPose/blob/main/utilities/networking.py
    """
    for url in urls:
        file_name = url.split('/')[-1].split('?')[0]

        print(f"Downloading file name : {file_name}")
 
        file_name = output_folder + '/' + file_name
        url = url[:-1] + '1'

        ## from contactpose
        tries = 0
        while tries < max_tries:
            done = download_url_once(url, file_name, True)
            if done:
                break
            else:
                t = 1
                print('*** Sleeping for {:f} s'.format(t))
                time.sleep(t)
                tries += 1
        if tries == max_tries:
            print('*** Download not complete. Max download tries exceeded')




def download_url_once(url, filename, progress=True):
    try:
        # headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        headers = {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}
        time.sleep(0.01)
        r = requests.get(url, stream=True, proxies=None, headers=headers)
    except ConnectionError as err:
        print(err)
        return False
    
    total_size = int(r.headers.get('content-length', 0))
    block_size = 102400 #100 Kibibyte
    if progress:
        t=tqdm(total=total_size, unit='iB', unit_scale=True)
    done = True
    datalen = 0
    with open(filename, 'wb') as f:
        itr = r.iter_content(block_size)
        while True:
            try:
                try:
                    data = next(itr)
                except StopIteration:
                    break
                if progress:
                    t.update(len(data))
                datalen += len(data)
                f.write(data)
            except KeyboardInterrupt:
                done = False
                print('Cancelled')
            except ConnectionError as err:
                done = False
                print(err)
    if progress:
        t.close()
    if (not done) or (total_size != 0 and datalen != total_size):
        print("ERROR, something went wrong")
        try:
            os.remove(filename)
        except OSError as e:
            print(e)
        return False
    else:
        return True




def extractBbox(hand_2d, image_rows=1080, image_cols=1920, bbox_w=640, bbox_h=480):
    # consider fixed size bbox
    x_min_ = min(hand_2d[:, 0])
    x_max_ = max(hand_2d[:, 0])
    y_min_ = min(hand_2d[:, 1])
    y_max_ = max(hand_2d[:, 1])

    x_avg = (x_min_ + x_max_) / 2
    y_avg = (y_min_ + y_max_) / 2

    x_min = max(0, x_avg - (bbox_w / 2))
    y_min = max(0, y_avg - (bbox_h / 2))

    if (x_min + bbox_w) > image_cols:
        x_min = image_cols - bbox_w
    if (y_min + bbox_h) > image_rows:
        y_min = image_rows - bbox_h

    bbox = [x_min, y_min, bbox_w, bbox_h]
    return bbox, [x_min_, x_max_, y_min_, y_max_]


# It's mine!
def extract_bbox_from_cropped(hand_2d, crop_box, resized_size=(640, 480)):
    """
    hand_2d: (N, 2) numpy array of keypoints in original image coordinates
    crop_box: [x_min_crop, y_min_crop, crop_w, crop_h] from original image
    resized_size: (width, height) of the resized image (default: 640x480)
    
    returns: bbox in resized image coordinate
    """
    x_min_crop, y_min_crop, crop_w, crop_h = crop_box
    target_w, target_h = resized_size

    # crop → resize の変換係数
    scale_x = target_w / crop_w
    scale_y = target_h / crop_h

    # キーポイントを crop → resize 後の座標に変換
    hand_2d_cropped = hand_2d.copy()
    hand_2d_cropped[:, 0] = (hand_2d[:, 0] - x_min_crop) * scale_x
    hand_2d_cropped[:, 1] = (hand_2d[:, 1] - y_min_crop) * scale_y

    # bbox生成
    x_min_ = min(hand_2d_cropped[:, 0])
    x_max_ = max(hand_2d_cropped[:, 0])
    y_min_ = min(hand_2d_cropped[:, 1])
    y_max_ = max(hand_2d_cropped[:, 1])

    bbox = [
        max(0, x_min_),
        max(0, y_min_),
        min(target_w, x_max_ - x_min_),
        min(target_h, y_max_ - y_min_)
    ]
    return bbox

import numpy as np
import trimesh
def to_4x4_matrix(matrix_3x4):
    """Convert a 3x4 matrix (as list or array) to a 4x4 homogenous matrix"""
    matrix_3x4 = np.array(matrix_3x4).reshape(3, 4)
    matrix_4x4 = np.eye(4)
    matrix_4x4[:3, :4] = matrix_3x4
    return matrix_4x4
def compute_object_bbox_2d(
    object_file,
    object_mat,
    extrinsic,
    intrinsic,
    crop_box,
    scale,
    resized_size=(640, 480)
):
    # to numpy if tensor
    if hasattr(extrinsic, 'cpu'):
        extrinsic = extrinsic.cpu().numpy()
    if hasattr(intrinsic, 'cpu'):
        intrinsic = intrinsic.cpu().numpy()

    # convert extrinsic to 4x4
    extrinsic = to_4x4_matrix(extrinsic)

    # スケール行列を生成して object_mat に掛ける
    if isinstance(scale, (int, float)):
        scale_matrix = np.diag([scale, scale, scale, 1.0])
    elif isinstance(scale, (list, tuple, np.ndarray)) and len(scale) == 3:
        scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    else:
        raise ValueError("Invalid scale format")
    
    object_mat = np.array(object_mat) @ scale_matrix

    # load object mesh
    mesh = trimesh.load(object_file, force='mesh')
    vertices = mesh.vertices
    vertices_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_world = (object_mat @ vertices_homo.T).T
    vertices_cam = (extrinsic @ vertices_world.T).T

    # projection (intrinsic)
    intrinsic_mat = intrinsic.reshape(3, 3)
    x, y, z = vertices_cam[:, 0], vertices_cam[:, 1], vertices_cam[:, 2] + 1e-5
    fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
    cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy

    # crop & resize transformation
    crop_x, crop_y, crop_w, crop_h = crop_box
    scale_x = resized_size[0] / crop_w
    scale_y = resized_size[1] / crop_h
    u_cropped = (u - crop_x) * scale_x
    v_cropped = (v - crop_y) * scale_y

    # bounding box in cropped/resized image
    x_min = max(0, np.min(u_cropped))
    y_min = max(0, np.min(v_cropped))
    x_max = min(resized_size[0], np.max(u_cropped))
    y_max = min(resized_size[1], np.max(v_cropped))

    return [int(x_min), int(y_min), int(x_max), int(y_max)]
