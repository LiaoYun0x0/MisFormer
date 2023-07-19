'''
Created on Jun 4, 2019

@author: ly
'''
'''
hsv means hue,saturation,value
hue means the color, red,blue,green,yellow etc.
saturation express the degree the color far from white.the bigger the saturation is, further from white,
and closer to a specified color.
value is the lightness of color, the bigger the value is, the lighter the color is.

normally, h in range(0,360), s and v in rang(0,1)
but in opencv, h in range(0,180) or range(0,360), s and v in range(0,256)
'''


import numpy as np
import cv2
import random


def change_lightness_contrast(image,alpha=0,beta=0):
    '''
    alpha in rnage(-1,1)
    beta in range(-255,255)
    '''
    transformed= np.clip(image*(1+alpha)+beta,0,255)
    return np.asarray(transformed, dtype=np.uint8)

def resize_pad_image(image,th=256,tw=256):
    h,w = image.shape[:2]
    scale = min(th/h,tw/w)
    n_image = np.ones((th,tw)) * 127.5
    if len(image.shape) == 3:
        n_image = np.tile(n_image[:,:,None],(1,1,image.shape[2]))
    r_image = cv2.resize(image, None,fx=scale,fy=scale)
    rh,rw = r_image.shape[:2]
    sh = (th - rh) // 2
    sw = (tw - rw) // 2
    n_image[sh:sh+rh,sw:sw+rw,...] = r_image
    return n_image

def random_normalize(image,min_scale=0.5):
    _mini = np.min(image)
    _maxi = np.max(image)
    max_scale = 500 / (_maxi - _mini)
    scale = np.random.rand(1)
    scale = (max_scale - min_scale) * scale + min_scale
    range_ = _maxi - _mini
    drange =  range_ * scale
    dx = (drange - range_) / 2
    alpha = int(np.clip(_mini - dx,0,255))
    beta = int(np.clip(_maxi + dx,0,255))
    alpha = _mini - dx
    beta = _maxi + dx
    image = cv2.normalize(image,None,alpha=alpha,beta=beta,norm_type=cv2.NORM_MINMAX)
    #print(scale,_mini,_maxi,alpha,beta,np.min(image),np.max(image))
    return image

def random_distort_hsv(image,h,s,v):
    '''
    all h,s,v are in range(-1,1)
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = np.clip(hsv[:,:,0]*(1+h),0,179)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*(1+s),0,255)
    hsv[:,:,2] = np.clip(hsv[:,:,2]*(1+v),0,255)
    hsv = np.asarray(hsv, dtype=np.uint8)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def blur_image(image,kernel_size=7):
    img = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    return img

def sharpen_image(image,kernel_size=7):
    img = blur_image(image, kernel_size)
    sharpen = cv2.addWeighted(image,2,img,-1,0)
    return sharpen

def random_crop(image,minimum_crop_rate=0.5,size=None):
    h,w = image.shape[:2]
    start_h = np.random.randint(0,int((1-minimum_crop_rate)/2*h))
    end_h = np.random.randint(int(start_h+h*minimum_crop_rate),h)
    start_w = np.random.randint(0,int((1-minimum_crop_rate)/2*w)) 
    end_w = np.random.randint(int(start_w+w*minimum_crop_rate),w)
    image = image[start_h:end_h,start_w:end_w,:]
    if size:
        image = cv2.resize(image, size)
    return image

def random_gauss_noise(image,mean=0,var=0.1):
    image = np.array(image/255, dtype=float)
    var_scale = np.random.rand(1)
    noise = np.random.normal(mean, var*var_scale, (image.shape[0],image.shape[1],1))
    noise = np.tile(noise,(1,1,3))
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out*255)
    return out

def random_mask(img,max_mask=30):
    h,w = img.shape[0:2]
    masks = np.random.randint(1,max_mask)
    for i in range(masks):
        start_h = np.random.randint(0,int(h*0.5))
        end_h = np.random.randint(start_h+10,h)
        start_w = np.random.randint(0,w-15)
        end_w = start_w + np.random.randint(3,10)
        val = np.random.randint(0,255)*np.ones((end_h-start_h,end_w-start_w,3))
        img[start_h:end_h,start_w:end_w,:] = val
    return img 
def random_transparent_mask(img,max_mask=20,min_width=32,max_width=100):
    h,w = img.shape[0:2]
    masks = np.random.randint(1,max_mask)
    for i in range(masks):
        start_h = np.random.randint(0,int(h*0.5))
        end_h = np.random.randint(start_h+10,h)
        start_w = np.random.randint(0,w-max_width)
        end_w = start_w + np.random.randint(min_width,max_width)
        val = np.random.randint(0,256)
        #mask_weight = np.random.rand(end_h-start_h,end_w-start_w,3)
        mask_weight = np.random.rand(1)/3
        img[start_h:end_h,start_w:end_w,...] = img[start_h:end_h,start_w:end_w,...]*(1-mask_weight)+val*mask_weight
    return np.asarray(img,dtype=np.uint8)

def draw_random_mask(img,num_mask=10,min_thick=32, max_thick=100):
    h,w = img.shape[:2]
    mask_img = np.zeros_like(img,np.uint8)
    x = np.random.randint(0,w,size=(num_mask))
    y = np.random.randint(0,h,size=(num_mask))

    for i in range(num_mask//2):
        thick = np.random.randint(min_thick,max_thick)
        b,g,r = np.random.randint(0,255,size=(3))
        cv2.line(mask_img,(x[2*i],y[2*i]),(x[2*i+1],y[2*i+1]),(int(b),int(g),int(r)),thick,cv2.FILLED)

    mask = mask_img > 0
    mask_weight =np.random.rand(1)/3
    img[mask] = img[mask] * (1-mask_weight) + mask_img[mask] * mask_weight
    return img
    
def random_rotation(img,max_degree=4,borderValue=None):
    h,w = img.shape[:2]
    degree = np.random.rand(1) * max_degree * 2 - max_degree
    matRotate = cv2.getRotationMatrix2D((h*0.5, w*0.5),float(degree),1)
    if borderValue is None:
        borderValue = np.random.randint(0,255)
    img = cv2.warpAffine(img, matRotate, (w,h),borderValue=borderValue)
    return img

def random_jitter(img,max_jitter=0.2,max_pixel=10):
    h,w = img.shape[:2]
    if max_pixel:
        hp,wp = max_pixel,max_pixel
        dh,dw = np.random.randint(0,2*max_pixel,size=(2)) - max_pixel
        sh = hp + dh
        sw = wp + dw
    else:
        h_jitter = np.random.rand(1) * max_jitter * 2 - max_jitter # range from (-max_jiiter, max_jitter)
        w_jitter = np.random.rand(1) * max_jitter * 2 - max_jitter
        hp = int(h*max_jitter + 0.5)
        wp = int(w*max_jitter + 0.5)
        sh = hp + int(h*h_jitter + 0.5)
        sw = wp + int(w*w_jitter + 0.5)
    nh = h + 2*hp
    nw = w + 2*wp
    if len(img.shape) == 3:
        array = np.ones((nh,nw,3)) * np.random.randint(0,256)
    else:
        array = np.ones((nh,nw)) * np.random.randint(0,256)
    array[hp:hp+h, wp:wp+w,...] = img
    return array[sh:sh+h, sw:sw+w,...]

def PepperandSalt(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.randint(0,src.shape[0]-1)
        randY=np.random.randint(0,src.shape[1]-1)
        if np.random.rand(1)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

def random_process(img):
    img = random_jitter(img)
    img = random_crop(image)
    img = random_transparent_mask(img)
    img = random_gauss_noise(img)
    img = blur_image(img, 3)
    alpha,h,v,s = np.random.rand(4)/2.5 - 0.2
    beta = np.random.randint(-50,50)
    img = change_lightness_contrast(img, alpha, beta)
    img = random_normalize(img)
    img = random_distort_hsv(img, h, s, v)
    return img



if __name__ == '__main__':
    
    o_img = cv2.imread("/home/ly/freedisk/Data/AerialImage/AerialImageDataset/train/images/austin1.tif",-1)
    while True:        
        img = o_img.copy()
        img = random_process(img)
        
        cv2.imshow('aug',img)
        q = cv2.waitKey(0) & 0xff
        if q == ord('q'):
            break
    cv2.destroyAllWindows()
