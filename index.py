import cv2
import numpy as np
from matplotlib import pyplot as plt
import utlis
import imutils


import random
import time




MIN_MATCH_COUNT = 10
# imgref = cv2.imread('9ac0d797d17b504d5df2430f523272a1.jpg')
# imgs = [cv2.imread('PicsArt_05-28-01.23.52.jpg'), cv2.imread('page09.jpg'), cv2.imread('Regularizar-CPF-nos-Correios-1280x720.jpg'), cv2.imread('example_2.jpg')]

# imgref = cv2.imread('Scanned/cpf antigo2.jpg') # queryImage
# imgs = [cv2.imread('example_2.jpg')]


# imgref = cv2.imread('Scanned/4457.jpg') # queryImage
# imgs = [cv2.imread('page09.jpg')]


imgref = cv2.imread('Scanned/cpf1.png') # queryImage
imgs = [cv2.imread('Regularizar-CPF-nos-Correios-1280x720.jpg')]


# imgref = cv2.imread('Scanned/cpf antigo2.jpg') # queryImage
# imgs = [cv2.imread('PicsArt_05-28-01.23.52.jpg')]


# imgref = cv2.imread('Scanned/4494.jpg') # queryImage
# imgs = [cv2.imread('9ac0d797d17b504d5df2430f523272a1.jpg')]



heightImg = 1000
widthImg  = 1500
print(cv2.__version__)


def featureMatching(img1, img2, applymatching = False, imgOriginal = None):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    print ("--------------------------------------------------")
    print ("Result good - %d/%d" % (len(good),MIN_MATCH_COUNT))

    if applymatching:

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h = img1.shape[0];
            w = img1.shape[1];

            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            imgOriginal = cv2.polylines(imgOriginal,[np.int32(dst)],True, (255,255, 255), 3, cv2.LINE_AA)

        else:
            print("---------------------------------------------")
            print ("Não possui a quantidade de correspondencia nescessaria - %d/%d" % (len(good),MIN_MATCH_COUNT))
            print("---------------------------------------------")
            matchesMask = None


        draw_params = dict(
            matchColor = (0,255,0), # draw matches in green color
            singlePointColor = None,
            matchesMask = matchesMask, # draw only inliers
            flags = 2
        );

        resultadoProcessing = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params);
        return [resultadoProcessing, imgOriginal]


    else:
        return len(good)



class ProccessImages(object):

    def __init__(self, img):
        self.img = img.copy()

    def naturalImage(self):
        return self.img.copy()

    def grayScaleImage(self):
        return cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY

    def blurImage(self):
        return cv2.blur(self.img.copy(), (6, 6))

    def gradientImg(self):
        #compute the Scharr gradient magnitude representation of the images
        #in both the x and y direction using OpenCV 2.4

        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F

        gradX = cv2.Sobel(self.img.copy(), ddepth=ddepth, dx=1, dy=0, ksize=0)
        gradY = cv2.Sobel(self.img.copy(), ddepth=ddepth, dx=0, dy=1, ksize=0)

        # subtract the y-gradient from the x-gradient

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        return gradient


    def __getattr__(self, attr):
        return None


def getArraysOfMethods(imgRef, imgAnalisis):

    arrayOfMethodsClass = [a for a in dir(ProccessImages) if not a.startswith('__')]

    imgsRefs = [];
    imagsAnalisis = [];

    imgRinit = ProccessImages(imgRef);
    imgAinit = ProccessImages(imgAnalisis.copy());

    for funcImagesProcess in arrayOfMethodsClass:

        imgr = getattr(imgRinit, funcImagesProcess)
        imgsRefs.append({
            "tecnica": funcImagesProcess,
            "resultado": imgr()
        });


        imga = getattr(imgAinit, funcImagesProcess);
        imagsAnalisis.append({
            "tecnica": funcImagesProcess,
            "resultado": imga()
        });


    return {
        "ImgsRefencia": imgsRefs,
        "ImgsAnalise": imagsAnalisis
    }




def identifyTecnicalApplyImage(imgRef, imgAnalisis):

    methods = getArraysOfMethods(imgRef.copy(), imgAnalisis.copy());

    goodMatching = 0;

    melhorMetodo = {};

    for imgReferencia in methods['ImgsRefencia']:

        for imgAnalise in methods['ImgsAnalise']:

            featmatching = featureMatching(imgReferencia['resultado'], imgAnalise['resultado'])
            if goodMatching < featmatching :

                goodMatching = featmatching;
                melhorMetodo = {
                    "metodoimgRef": imgReferencia['tecnica'],
                    "metodoimgAnalis": imgAnalise['tecnica'],
                    "imgRef": imgReferencia['resultado'],
                    "imgResult": imgAnalise['resultado'],
                    "resultMatchin": goodMatching
                };

    return melhorMetodo;






def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()




def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area







def initanalisesimage():

    for img in imgs:

        resized = imutils.resize(img.copy(), width=300)

        print ("Altura (height): %d pixels" % (resized.shape[0]))
        print ("Largura (width): %d pixels" % (resized.shape[1]))
        print ("Canais (channels): %d"      % (resized.shape[2]))

        metodo = identifyTecnicalApplyImage(imgref.copy(), resized.copy());

        img1 = metodo['imgRef'];
        img2 = metodo['imgResult'];

        imgresult, image2alterada = featureMatching(img1.copy(), img2.copy(), True, img.copy())

        initProcessclass = ProccessImages(image2alterada.copy())
        imgGray = initProcessclass.grayScaleImage();

        imgEdg = cv2.Canny(imgGray.copy(),500,500) # APPLY CANNY BLUR

        (_, thresh) = cv2.threshold(imgGray.copy(), 254, 255, cv2.THRESH_BINARY)


        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

        # # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)

        # draw a bounding box arounded the detected barcode and display the
        # image
        draw = cv2.drawContours(resized.copy(), [box], -1, (0, 255, 0), 10)


        biggest, maxArea = biggestContour([box]) # FIND THE BIGGEST CONTOUR

        if biggest.size != 0:
            biggest=utlis.reorder(biggest)
            cv2.drawContours(draw.copy(), biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
            imgBigContour = utlis.drawRectangle(draw.copy(),biggest,2)
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(draw.copy(), matrix, (widthImg, heightImg))

            #REMOVE 20 PIXELS FORM EACH SIDE
            imgWarpColored= imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

            # APPLY ADAPTIVE THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored.copy(),cv2.COLOR_BGR2GRAY)

            imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray.copy(),255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

            # Image Array for Display
            images = [imgref, img, img1, img2, imgresult, image2alterada, imgGray, imgEdg, thresh, draw, imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre]
            titles = ["imgRef", "img", "img1", "img2", "imgResult", "image2alterada", "imgGray", "imgEdg", "thresh", "draw", "BigCountor", "Warp", "WarpGray", "Adaptative"]
            show_images(images, 4, titles)

        else:

            images = [imgref, img, img1, img2, imgresult, image2alterada, imgGray, imgEdg, thresh, draw]
            titles = ["imgRef", "img", "img1", "img2", "imgResult", "image2alterada", "imgGray", "imgEdg", "thresh", "draw"]
            show_images(images, 3, titles)




initanalisesimage();
