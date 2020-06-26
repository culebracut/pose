import torch
import torch2trt
import trt_pose.models
import os
import subprocess
import tensorrt as trt


MODEL = trt_pose.models.densenet121_baseline_att
WEIGHTS = 'https://nvidia.box.com/shared/static/mn7f3a8le9bn8cwihl0v6s9wlm5damaq.pth'
OUTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'generated')
TORCH_WEIGHTS_PATH = os.path.join(OUTDIR, 'densenet121_baseline_att.pth')
TRT_WEIGHTS_PATH = os.path.join(OUTDIR, 'densenet121_baseline_att_trt.pth')


if __name__ == '__main__':

    # download weights
    subprocess.call(['wget', WEIGHTS, '-O', TORCH_WEIGHTS_PATH])

    # load weights
    model = MODEL(18, 42).cuda().eval()
    model.load_state_dict(torch.load(TORCH_WEIGHTS_PATH))

    # optimize
    data = torch.randn((1, 3, 224, 224)).cuda().float()
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25, log_level=trt.Logger.VERBOSE)

    # save
    torch.save(model_trt.state_dict(), TRT_WEIGHTS_PATH)
