import cv2, sys
sys.path.append("./realTimeStyleTransferSlim/")
from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
import numpy as np
import pathlib, os
from realTimeStyle import realTimeStyle


localPath = str(pathlib.Path(__file__).parent.absolute())

class styleTransferViz(object):
    def __init__(self, stylesList):

        self.baseGraphicsPath = os.path.join(localPath, "graphics/")

        # Loading base images
        self.baseFrame      = cv2.imread(os.path.join(localPath, "graphics/styleTransferBase.png"), -1)
        self.alchemist      = cv2.imread(os.path.join(localPath, "graphics/styleTransferAlchemist.png"), -1)
        self.alchemistPiece = cv2.imread(os.path.join(localPath, "graphics/styleTransferAlchemistPiece.png"), -1)

        # Initialize main window with default background
        self.outputWinName = "Game Painter"
        self.initWindow(windowName=self.outputWinName)
        self.show(self.baseFrame, windowName=self.outputWinName, wait=2000)

        self.baseFrameDim      = self.baseFrame.shape[:2] # H, W tuple
        self.alchemistDim      = self.alchemist.shape[:2] # H, W tuple
        self.alchemistPieceDim = self.alchemistPiece.shape[:2] # H, W tuple
        self.alchemistPieceMap = self.alchemistPiece[:,:,3]/255
        self.alchemistPieceMap = np.expand_dims(self.alchemistPieceMap, axis = 2)

        # Styles list
        self.stylesList = stylesList
        self.styleIdx = 0

        # Font management
        self.fontPath = localPath + "/graphics/Orbitron-Regular.ttf"

        self.fontSizeXL = 30
        self.fontXL = ImageFont.truetype(self.fontPath, self.fontSizeXL)
        self.fontSizeL = 46
        self.fontL = ImageFont.truetype(self.fontPath, self.fontSizeL)
        self.fontSizeM = 30
        self.fontM = ImageFont.truetype(self.fontPath, self.fontSizeM)
        self.fontSizeS = 24
        self.fontS = ImageFont.truetype(self.fontPath, self.fontSizeS)
        self.fontSizeXS = 20
        self.fontXS = ImageFont.truetype(self.fontPath, self.fontSizeXS)
        self.fontSizeXXS = 18
        self.fontXXS = ImageFont.truetype(self.fontPath, self.fontSizeXXS)
        self.fontSizeXXXS = 16
        self.fontXXXS = ImageFont.truetype(self.fontPath, self.fontSizeXXXS)

        # Color management
        self.colorYell = (255, 200, 0)
        self.colorYellInv = (0, 200, 255)

        self.space = 10

        self.counter = 0

        # Max height for style
        self.maxStyleHeight = 500

        # Game Frame
        self.gameFrameDim = (590, 768)

        # Positions initialization
        self.initElemPos()

        # Initialize Style
        self.loadStyle()

    # Init text/graphic positions
    def initElemPos(self):

        self.styleImgPosRightLim = (525, 1230)

        self.styleImgTextPos = (1000, 1250)

        self.alchemistPos = (0, 750)

        self.processedFramePos = (218, 1380)
        self.originalFramePos = (100, 80)

    # Style image and description load
    def loadStyle(self):

        styleDict = self.stylesList[self.styleIdx]

        self.styleDescription = styleDict["name"]
        print("Loading \"{}\" style".format(self.styleDescription))

        self.styleImg    = cv2.imread(styleDict["styleImg"], -1)
        if self.styleImg.shape[2] == 3:
            # First create the image with alpha channel
            self.styleImg = cv2.cvtColor(self.styleImg, cv2.COLOR_RGB2RGBA)

            # Then assign the mask to the last channel of the image
            self.styleImg[:,:,3] = np.ones((self.styleImg.shape[0], self.styleImg.shape[1]))*255

        self.styleImgDim = (self.maxStyleHeight,
                            int(self.styleImg.shape[1]*self.maxStyleHeight/self.styleImg.shape[0]))
        self.styleImg = cv2.resize(self.styleImg, (self.styleImgDim[1], self.styleImgDim[0]),
                                   interpolation = cv2.INTER_AREA)

        # Fuse style and Alchemist
        self.fuseStyleAndAlchemist()

        # Add elements to base frame
        self.baseFrame = cv2.imread(localPath + "/graphics/styleTransferBase.png", -1)

        # Add fuse style and Alchemist
        self.baseFrame[self.styleAndAlchemistPos[0]:self.styleAndAlchemistPos[0]+self.styleAndAlchemistDim[0],
                       self.styleAndAlchemistPos[1]:self.styleAndAlchemistPos[1]+self.styleAndAlchemistDim[1],0:3] =\
                          self.styleAndAlchemist[:,:,0:3]

        # Adding text
        im_pil = Image.fromarray(self.baseFrame)
        self.addText( im_pil, self.styleDescription, self.styleImgTextPos, self.fontM,
                      self.fontSizeM, color=self.colorYellInv)
        self.baseFrame = np.asarray(im_pil)

        # Loading deep network
        self.stylizer = realTimeStyle(styleDict["model"], img_wh=(300, 200))

        #cv2.imwrite("./testFuse.png", self.styleAndAlchemist)
        #cv2.imwrite("./testBase.png", self.baseFrame)

    # Style image and description load
    def nextStyle(self):

        self.styleIdx = (self.styleIdx + 1) % len(self.stylesList)
        self.loadStyle()

    # Style and alchemist fusion
    def fuseStyleAndAlchemist(self):

        overlapWidth = min(self.styleImgDim[1], self.styleImgPosRightLim[1] - self.alchemistPos[1])
        self.startingWidth = max(self.styleImgPosRightLim[1] - self.alchemistPos[1] - self.styleImgDim[1], 0)
        self.fusedImgDim = (self.styleImgPosRightLim[0] + self.styleImgDim[0],
                            self.styleImgDim[1] + self.alchemistDim[1] - overlapWidth, 4)

        mapImage = self.alchemist[:,:,3]/255 # > 0
        mapImage = np.expand_dims(mapImage, axis = 2)

        self.styleAndAlchemist = np.zeros(self.fusedImgDim)
        self.styleAndAlchemist[0:self.styleImgDim[0], self.startingWidth:self.startingWidth+ self.styleImgDim[1],:] =\
           self.styleImg
        self.styleAndAlchemist[-self.alchemistDim[0]:, -self.alchemistDim[1]:, :] =\
            self.styleAndAlchemist[-self.alchemistDim[0]:, -self.alchemistDim[1]:, :]*(1-mapImage) +\
                self.alchemist*mapImage

        # Alchemist and style info
        self.styleAndAlchemistDim = self.styleAndAlchemist.shape
        self.styleAndAlchemistPos = (30, min(self.alchemistPos[1], self.styleImgPosRightLim[1]-self.styleImgDim[1]))
        self.mapImageFused = self.styleAndAlchemist[:,:,3]/255
        self.mapImageFused = np.expand_dims(self.mapImageFused, axis = 2)

    # Read image
    def readImage(self, path, dims=None):

        if dims == None:
            return cv2.imread(path, -1)
        else:
            return cv2.resize(cv2.imread(path, -1),
                             (dims[0], dims[1]), interpolation = cv2.INTER_AREA)

    def addText(self, base, text, pos, fontIn, fontInSize, color=(0,0,200), rightAlign=False, shift=5):

        baseShape = base.size
        posLoc = [0, 0]
        posLoc[1] = baseShape[1] - pos[0] - fontInSize + shift
        posLoc[0] = pos[1]

        draw = ImageDraw.Draw(base)
        w = 0
        if rightAlign:
            w, h = draw.textsize(text, font=fontIn)
        draw.text((posLoc[0]-w, posLoc[1]), text, color, font=fontIn)

    def addImage(self, base, image, pos):

        imgShape = image.shape[:3]
        posLoc = [0, 0]
        posLoc[0] = self.baseFrameDim[0] - pos[1] - image.shape[0]
        posLoc[1] = pos[0]

        mapImage = image[:,:,3]/255 # > 0
        mapImage = np.expand_dims(mapImage, axis = 2)

        base[posLoc[0]:posLoc[0]+imgShape[0], posLoc[1]:posLoc[1]+imgShape[1], :] =\
          base[posLoc[0]:posLoc[0]+imgShape[0], posLoc[1]:posLoc[1]+imgShape[1], :]*(1-mapImage) +\
          image*mapImage

    def addImageNoAlpha(self, base, image, pos):

        imgShape = image.shape[:3]
        posLoc = [0, 0]
        posLoc[0] = self.baseFrameDim[0] - pos[1] - image.shape[0]
        posLoc[1] = pos[0]

        base[posLoc[0]:posLoc[0]+imgShape[0], posLoc[1]:posLoc[1]+imgShape[1], :] = image

    def initWindow(self, windowName=None):

        cv2.namedWindow(windowName,cv2.WINDOW_GUI_NORMAL)
        #cv2.resizeWindow(windowName, 1352,1014)
        cv2.resizeWindow(windowName, 2560,1055)
        cv2.moveWindow(windowName, 0,0)

    def show(self, output, windowName=None, wait=0):
        #output = cv2.resize(output, (1352,1014), interpolation = cv2.INTER_AREA)

        cv2.imshow(windowName, output)

        cv2.waitKey(wait) & 0xFF

    def styleGame(self, frame, waitPress=0):

        output = self.baseFrame.copy()

        # Add original frame to output
        originalFrame = cv2.resize(frame, (self.gameFrameDim[1], self.gameFrameDim[0]), interpolation = cv2.INTER_AREA)
        output[-self.originalFramePos[0]-self.gameFrameDim[0]:-self.originalFramePos[0],
                self.originalFramePos[1]:self.originalFramePos[1]+self.gameFrameDim[1], 0:3] = originalFrame[:,:,::-1]

        # Style the game frame with the model
        frame = self.stylizer.styleFrame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resizing frame to fixed dim
        frame = cv2.resize(frame, (self.gameFrameDim[1], self.gameFrameDim[0]), interpolation = cv2.INTER_AREA)

        # Add alchemist piece to frame
        frame[-self.alchemistPieceDim[0]:,:self.alchemistPieceDim[1], :] =\
            frame[-self.alchemistPieceDim[0]:,:self.alchemistPieceDim[1], :]*(1-self.alchemistPieceMap) +\
            self.alchemistPiece[:,:,0:3]*self.alchemistPieceMap

        # Add frame to base frame
        output[-self.processedFramePos[0]-self.gameFrameDim[0]:-self.processedFramePos[0],
                self.processedFramePos[1]:self.processedFramePos[1]+self.gameFrameDim[1], 0:3] =\
            frame[:,:,:]

        # Add style and alchemist to base frame
        #output[self.styleAndAlchemistPos[0]:, self.styleAndAlchemistPos[1]:self.styleAndAlchemistPos[1]+\
        #       self.styleAndAlchemistDim[1], :] =\
        #       output[self.styleAndAlchemistPos[0]:, self.styleAndAlchemistPos[1]:self.styleAndAlchemistPos[1]+\
        #              self.styleAndAlchemistDim[1], :]*(1-self.mapImageFused) +\
        #                  self.styleAndAlchemist*self.mapImageFused

        self.show(output, windowName=self.outputWinName, wait=waitPress)

    def closeAll(self):
        cv2.destroyAllWindows()
