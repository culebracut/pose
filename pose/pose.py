import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import torch2trt
import trt_pose.coco
from trt_pose.parse_objects import ParseObjects

COCO_CATEGORY = {
        "supercategory": "person", 
        "id": 1, 
        "name": "person", 
        "keypoints": [
            "nose", 
            "left_eye", 
            "right_eye", 
            "left_ear", 
            "right_ear", 
            "left_shoulder", 
            "right_shoulder", 
            "left_elbow", 
            "right_elbow", 
            "left_wrist", 
            "right_wrist", 
            "left_hip", 
            "right_hip", 
            "left_knee", 
            "right_knee", 
            "left_ankle", 
            "right_ankle", 
            "neck"
        ], 
        "skeleton": [
            [16, 14], 
            [14, 12], 
            [17, 15], 
            [15, 13], 
            [12, 13], 
            [6, 8], 
            [7, 9], 
            [8, 10], 
            [9, 11], 
            [2, 3], 
            [1, 2], 
            [1, 3], 
            [2, 4], 
            [3, 5], 
            [4, 6], 
            [5, 7], 
            [18, 1], 
            [18, 6], 
            [18, 7], 
            [18, 12], 
            [18, 13]
        ]
}


TOPOLOGY = trt_pose.coco.coco_category_to_topology(COCO_CATEGORY)


class PosePre(object):
    
    def __init__(self, shape=(224, 224), dtype=torch.float32, device=torch.device('cuda')):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).type(dtype)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).type(dtype)
        
    def __call__(self, image):
        with torch.no_grad():
            image = cv2.resize(image, self.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            image = transforms.functional.to_tensor(image).to(self.device).type(self.dtype)
            image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            image = image[None, ...]
        return image
 

class PoseEngine(object):

    def __init__(self, path):
        self.module = torch2trt.TRTModule()
        self.module.load_state_dict(torch.load(path))


    def __call__(self, tensor):
        cmap, paf = self.module(tensor)
        return cmap, paf


class PosePost(object):
    
    def __init__(self, *args, **kwargs):
        self.parse_objects = ParseObjects(TOPOLOGY, *args, **kwargs)
        
    def __call__(self, cmap, paf):
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        return counts, objects, peaks
 

class PoseDraw(object):

    def __init__(self, joint_color=(0, 255, 0), link_color=(100, 100, 100)):
        self.joint_color = joint_color
        self.link_color = link_color
    
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = TOPOLOGY
        joint_color = self.joint_color
        link_color = self.link_color
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])
        for i in range(count):
            
            obj = objects[0][i]
            # filter no-neck
            if obj[-1] < 0:
                continue
                

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), link_color, 2)
                    
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, joint_color, -1)
    


