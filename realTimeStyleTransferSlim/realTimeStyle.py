import argparse
import cv2
import torch

import numpy as np
#import onnxruntime as rt

from network import normal
from network import slim

class realTimeStyle:

    def __init__(self,  model_path, img_wh=(512, 512), model_mode="slim", is_onnx=False):

        if not torch.cuda.is_available():
            print("ERROR: cuda is not available")
            sys.exit(1)

        self.device = torch.device("cuda")
        self.modelPath = model_path
        self.imgWh     = img_wh
        self.modelMode = model_mode
        self.isOnnx   = is_onnx

        # define scale factor
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.stdResh = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        self.meanResh = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))

        if self.isOnnx:
            self.sess = rt.InferenceSession(self.modelPath)
            self.input_name = sess.get_inputs()[0].name
        else:
            # define network
            if self.modelMode != "slim":
                self.model = normal.ImageTransformNet(self.imgWh).to(self.device)
            else:
                self.model = slim.ImageTransformNet().to(self.device)
            self.model.load_state_dict(torch.load(self.modelPath))
            self.model.eval()

    def preProcessFrame(self, img):

        # load image
        #img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, self.imgWh)
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        for i in range(3):
            img[i] -= self.mean[i]
            img[i] /= self.std[i]

        return img

    def styleFrame(self, img):

        img = self.preProcessFrame(img)

        # ndarray to tensor
        img = torch.from_numpy(np.array([img])).float().to(self.device)

        # inference
        out = self.model(img).cpu()

        out = out.detach().numpy()
        out = np.squeeze(out)
        out = ((out * self.stdResh + self.meanResh).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")

        return out

    def styleFrameOnnx(self, img):
        out = self.sess.run(None, {self.input_name: img.astype(np.float32)})[0]

        return out

    def showStyledFrameprocess(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("DIAMBRA Style", img)
        cv2.waitKey(0)
