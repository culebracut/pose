import time
import argparse
from pose import PosePre, PoseEngine, PosePost, PoseDraw
import sys
import os
import cv2

UTILS_PATH = '{}/workspace/pose/trt_pose/utils/'.format(os.environ.get('HOME'))
ENGINE_PATH = '{}/workspace/pose/pose/generated/densenet121_baseline_att_trt.pth'.format(os.environ.get('HOME'))
sys.path.append(UTILS_PATH)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=270)
    parser.add_argument('--codec', type=str, default='h264')
    args = parser.parse_args()
    width = args.width
    height = args.height

    # media
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Display")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # pose
    pre = PosePre()
    engine = PoseEngine(ENGINE_PATH)
    post = PosePost()
    draw = PoseDraw()

    t0 = time.time()

    while True:

        # read image from file
        """         image = video.read() """
        ret, image = cap.read()

        # preprocess image (resize, send to GPU, normalize, permute dims)
        tensor = pre(image)

        # run pose engine
        cmap, paf = engine(tensor)

        # parse objects
        counts, objects, peaks = post(cmap, paf)

        # draw objects on image
        draw(image, counts, objects, peaks)

        # render image to display
        cv2.imshow('Display',image)

        k = cv2.waitKey(1)
        if k == 27:         # ESC key to exit
            break

        # print FPS
        t1 = time.time()
        print('FPS: %f' % (1.0 / (t1 - t0)))
        t0 = t1
    
    # exit
    cv2.destroyAllWindows()
