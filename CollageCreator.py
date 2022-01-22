import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import numpy as np
import os
from skimage import feature #for canny edge detector
from skimage.color import rgb2gray

IMG_WIDTH=639 #size divisible by 3
IMG_HEIGHT=480 #divisible by 2
IMG_SIZE=(IMG_HEIGHT,IMG_WIDTH)
COLLAGE_SIZE=(int(IMG_HEIGHT/2),int(IMG_WIDTH/3)) #every image size in collage of 3x2 grid


def blur_Intersection(img,ih,iw,H,W): #filter for bluring just intersecting boundary region (using 5X5 Gaussian filter)
    if (ih<0 or ih>=H or iw<0 or iw>=W):
        return
    filter_size=(5,5)
    for k in range(3): #for 3 channels rgb
        x=np.zeros(filter_size)
        for i in range(ih-2,ih+3):
            if (i<0 or i>=H):
                continue
            for j in range(iw-2,iw+3):
                if (j<0 or j>=W):
                    continue
                x[i-ih+2,j-iw+2]=img[i,j,k]
        y=np.array([  
            [1,4,7,4,1],
            [4,16,26,16,4],
            [7,26,41,26,7],
            [4,16,26,16,4],
            [1,4,7,4,1]],np.float_) #5x5 Gaussian filter
        x=x*y
        img[ih,iw,k]=sum(sum(x))/(273)

#### HOW COLLAGE I HAVE MERGED #####
def rawEdgeCollage(edges,pixel_count,height,width):#How raw canny edge detected images merged to form collage
    eoutputImage=np.zeros([height,width])
    for iH in range(0,height):
        for iW in range(0,width):
            img_index=int(iH/COLLAGE_SIZE[0])*3 + int(iW/COLLAGE_SIZE[1])            
            h=iH%COLLAGE_SIZE[0]
            w=iW%COLLAGE_SIZE[1]
            eoutputImage[iH,iW]=edges[pixel_count[img_index][1]][h,w]
            
    for i in range(height):
        j=int(width/3) 
        eoutputImage[i,j]=1
        j+=j
        eoutputImage[i,j]=1

    for j in range(width):
        i=int(height/2) 
        eoutputImage[i,j]=1   
    
    plt.title("Collage of the canny edge detector images sorted on the basis of set pixel count of binary image")
    # plt.figure(figsize =(20,20))
    plt.imshow(eoutputImage, cmap='gray')
    plt.axis('off')
    plt.show()

def CollageCreate(path):
    # path="frames2" #relative path of folder which contain further contains frames
    
    ## READING IMAGES FROM GIVEN FOLDER PATH ##
    files=os.listdir(path)
    files=files[:6] #Need only six images
    images=[] #Array of ready images (in rgb format)
    for file in files:
        img=io.imread(os.path.join(path,file))
        img=resize(img,COLLAGE_SIZE)
        images.append(img)
    
    
    ####RESIZING IMAGES TO 639x480 SIZE ########

    height,width,depth=(IMG_HEIGHT,IMG_WIDTH,3)
    outputImage=np.zeros([height,width,depth])
    gimages=[rgb2gray(i) for i in images] #Grayscale images for edges detection

    edges=[feature.canny(i, sigma=3) for i in gimages] #Applying canny edge detector with higher sigma value 
                                                        # Higher sigma value will help to detect meandingfull edges


    ######  IMAGES SORTING #####
    pixel_count=[] #It will keep number of the count of the number of pixels which are part of edges
             # (Images is binary now, hence we can count number of pixels in image with are true)
             # It also keep the index of the image.
    for i in range(len(edges)):
        pixel_count.append([np.count_nonzero(edges[i]),i])
    
    pixel_count.sort() #Images are sorted on the basis number of edges boundaries detected by canny edge detector in an image
                # All images are of equal size
    for i in range(len(pixel_count)):
        print("Image at location ",i+1, " in grid have #",pixel_count[i][0]," set pixels")


    #### MERGING IMAGES TO FINAL COLLAGE #####
    for iH in range(0,height):
        for iW in range(0,width):
            img_index=int(iH/COLLAGE_SIZE[0])*3 + int(iW/COLLAGE_SIZE[1])
            for iCh in range(0,depth):            
                h=iH%COLLAGE_SIZE[0]
                w=iW%COLLAGE_SIZE[1]
                
                outputImage[iH,iW,iCh]=images[pixel_count[img_index][1]][h,w,iCh] #images picked on the basis of sorted pixels


    ##### BLURRING THE INTERSECTION BOUNDARIES OF FINAL COLLAGE ######
    blur_boundary=15 #For blurring the neighbour pixels at the intersection of images in collage
    
    for i in range(IMG_HEIGHT):
            cpixel=COLLAGE_SIZE[1]*1
            for k in range(cpixel-blur_boundary,cpixel+blur_boundary):
                blur_Intersection(outputImage,i,k,IMG_HEIGHT,IMG_WIDTH)
                
            cpixel=COLLAGE_SIZE[1]*2
            for k in range(cpixel-blur_boundary,cpixel+blur_boundary):
                blur_Intersection(outputImage,i,k,IMG_HEIGHT,IMG_WIDTH)

    cpixel=COLLAGE_SIZE[0]
    for i in range(cpixel-blur_boundary,cpixel+blur_boundary):
        for k in range(IMG_WIDTH):
            blur_Intersection(outputImage,i,k,IMG_HEIGHT,IMG_WIDTH)
    
    # plt.figure(figsize =(40,40))
    plt.imshow(outputImage)
    plt.axis('off')
    plt.show()

    #### HOW COLLAGE I HAVE MERGED #####
    rawEdgeCollage(edges,pixel_count,height,width)
