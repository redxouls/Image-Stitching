import argparse
import os
import sys
import time
import math
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import cmp_to_key
from numpy.linalg import lstsq, norm

SIFT_INIT_SIGMA = 0.5
SIFT_ORI_HIST_BINS = 36
SIFT_ORI_SIG_FCTR = 1.5
SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
SIFT_ORI_PEAK_RATIO = 0.8
SIFT_FIXPT_SCALE = 1
SIFT_DESCR_HIST_BINS = 8
SIFT_DESCR_WIDTH = 4
SIFT_DESCR_SCL_FCTR = 3
SIFT_DESCR_MAG_THR = 0.2
SIFT_INT_DESCR_FCTR = 512


class SIFT:
    def __init__(self):
        pass

    # create base image with double size and convert rgb to gray
    @staticmethod
    def create_initial_image(img, sigma):
        print(f"Shape of input image: {str(img.shape)}")
        print("Creating base image...")
        if(len(img.shape) > 2 and (img.shape[2] == 3 or img.shape[2] == 4)):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        sig_diff = math.sqrt(max(sigma**2 - (2*SIFT_INIT_SIGMA)**2, 0.01) )
        dbl = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        dbl = cv2.GaussianBlur(dbl, (0, 0), sigmaX=sig_diff, sigmaY=sig_diff)
        return dbl

    # build a gussian pyramid from base image with given sigma and n_octave_layers
    @staticmethod
    def build_gaussian_pyramid(img, sigma, n_octave_layers):
        print("Building Gaussian pyramid...")
        n_octaves = int(np.round(np.log(min(img.shape)) / np.log(2) - 1))
        
        n_images_per_octave = n_octave_layers + 3
        sig = np.zeros(n_images_per_octave)

        k = pow(2, 1 / n_octave_layers)
        #  precompute Gaussian sigmas using the following formula:
        #  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
        sig[0] = sigma
        
        for i in range(n_images_per_octave):
            sig_prev = pow(k, i-1) * sigma
            sig_total = sig_prev * k
            sig[i] = math.sqrt(sig_total**2 - sig_prev**2)

        # building gussian pyramid
        dim = (n_octaves, n_images_per_octave)
        pyr = np.empty(dim, dtype=object)
        
        for o in range(n_octaves):
            pyr[o][0] = img
            for i in range(1, n_images_per_octave):
                img = cv2.GaussianBlur(img, (0, 0), sig[i], sig[i])
                pyr[o][i] = img

            octave_base = pyr[o,-3]
            img = cv2.resize(octave_base, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        
        print("Finsihed building Gaussian pyramid of shape: %s" % str(pyr.shape))
        return pyr

    @staticmethod
    def build_DoG_pyramid(pyr):
        dim = (pyr.shape[0], pyr.shape[1]-1)
        dog_pyramid  = np.empty(dim, dtype=object)

        for i in range(dim[0]):
            for j in range(dim[1]):
                dog_pyramid[i, j] = np.array(cv2.subtract(pyr[i, j+1], pyr[i, j]), dtype=np.float32)
        
        return dog_pyramid
    
    @staticmethod
    def find_scale_space_extrema(pyr, dog_pyr,  n_octave_layers, sigma, image_border_width, contrast_threshold=0.04):
        print('Finding scale-space extrema...')
        n_octaves = pyr.shape[0]
        threshold = np.floor(0.5 * contrast_threshold / n_octave_layers * 255 * SIFT_FIXPT_SCALE)  # from OpenCV implementation
        
        n = SIFT_ORI_HIST_BINS
        keypoints = []

        for o in range(n_octaves):
            for idx in range(1, n_octave_layers+1):
                img = dog_pyr[o, idx]
                prev_img = dog_pyr[o, idx-1]
                next_img = dog_pyr[o, idx+1]
                
                # r: stands for row
                # c: stands for column
                for r in tqdm(range(image_border_width, prev_img.shape[0] - image_border_width)):
                    for c in range(image_border_width, prev_img.shape[1] - image_border_width):
                        if SIFT.is_extremum(img[r-1:r+2, c-1:c+2], prev_img[r-1:r+2, c-1:c+2], next_img[r-1:r+2, c-1:c+2], threshold):
                            localization_result = SIFT.localize_extremum_via_quadratic_fit(r, c, idx, o, n_octave_layers, dog_pyr, sigma, contrast_threshold, image_border_width)
                            if localization_result != False:
                                kpt, r1, c1, layer = localization_result
                                scl_octv = kpt.size*0.5/(1 << o)
                                
                                omax, hist = SIFT.calc_orientation_hist(pyr[o, layer], kpt, o, int(np.round(SIFT_ORI_RADIUS * scl_octv)), SIFT_ORI_SIG_FCTR * scl_octv, n)
                                mag_thr = omax * SIFT_ORI_PEAK_RATIO
                
                                for j in range(n):
                                    l = (j - 1) % n
                                    r2 = (j + 1 ) % n
                                
                                    if(hist[j] > hist[l] and hist[j] > hist[r2] and hist[j] >= mag_thr):
                                        bin_vote = (j + 0.5 * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2])) % n
                                        kpt.angle = 360 - (float)((360/n) * bin_vote)
                                        if np.zeros_like(abs(kpt.angle - 360)):
                                            kpt.angle = 0
                                        keypoints.append(kpt)
        return keypoints
    @staticmethod
    def is_extremum(img_sub, prev_img_sub, next_img_sub, threshold):
        val = img_sub[1, 1]
        if abs(val) > threshold:
            if val > 0:
                return (np.all(val >= prev_img_sub) and np.all(val >= next_img_sub) and np.all(val >= img_sub))
            elif val < 0:
                return (np.all(val <= prev_img_sub) and np.all(val <= next_img_sub) and  np.all(val <= img_sub))
        return False


    @staticmethod
    def localize_extremum_via_quadratic_fit(r, c, image_index, octave_index, n_octave_layers, dog_pyr, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, max_interp_steps=5):
        img_scale = 1/(255*SIFT_FIXPT_SCALE)
        deriv_scale = img_scale*0.5
        second_deriv_scale = img_scale
        cross_deriv_scale = img_scale*0.25
        
        for i in range(max_interp_steps):
            img = dog_pyr[octave_index, image_index]
            prev_img = dog_pyr[octave_index, image_index-1]
            next_img = dog_pyr[octave_index, image_index+1]

            dD = np.array([(img[r, c+1] - img[r, c-1]), (img[r+1, c] - img[r-1, c]), (next_img[r, c] - prev_img[r, c])]) * deriv_scale

            v2 = img[r, c]*2
            
            dxx = (img[r, c+1] + img[r, c-1] - v2)*second_deriv_scale
            dyy = (img[r+1, c] + img[r-1, c] - v2)*second_deriv_scale
            dss = (next_img[r, c] + prev_img[r, c] - v2)*second_deriv_scale
            
            dxy = (img[r+1, c+1] - img[r+1, c-1] - img[r-1, c+1] + img[r-1, c-1])*cross_deriv_scale
            dxs = (next_img[r, c+1] - next_img[r, c-1] - prev_img[r, c+1] + prev_img[r, c-1])*cross_deriv_scale
            dys = (next_img[r+1, c] - next_img[r-1, c] - prev_img[r+1, c] + prev_img[r-1, c])*cross_deriv_scale

            H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
            X = lstsq(H, dD, rcond=None)[0]
            
            xc, xr, xi = tuple(-X)
            
            if np.all(np.abs(X)<0.5):
                break
            
            c += int(np.round(xc))
            r += int(np.round(xr))
            image_index += int(np.round(xi))

            if(image_index < 1 or image_index > n_octave_layers or
                c < image_border_width or c >= img.shape[1] - image_border_width  or
                r < image_border_width or r >= img.shape[0] - image_border_width):
                return False
        

        # ensure convergence of interpolation
        if i >= max_interp_steps:
            return False
        else:
            img = dog_pyr[octave_index, image_index]
            prev_img = dog_pyr[octave_index, image_index-1]
            next_img = dog_pyr[octave_index, image_index+1]

            dD = np.array([(img[r, c+1] - img[r, c-1]), (img[r+1, c] - img[r-1, c]), (next_img[r, c] - prev_img[r, c])]) * deriv_scale
            
            t = dD.dot(-X)

            contr = img[r, c]*img_scale + t * 0.5
            if(abs(contr) * n_octave_layers < contrast_threshold):
                return False

            # principal curvatures are computed using the trace and det of Hessian
            v2 = img[r, c]*2
            dxx = (img[r, c+1] + img[r, c-1] - v2)*second_deriv_scale
            dyy = (img[r+1, c] + img[r-1, c] - v2)*second_deriv_scale
            dxy = (img[r+1, c+1] - img[r+1, c-1] - img[r-1, c+1] + img[r-1, c-1])*cross_deriv_scale
            
            tr = dxx + dyy
            det = dxx * dyy - dxy * dxy

            if(det <= 0 or (tr**2)*eigenvalue_ratio >= ((eigenvalue_ratio + 1)**2)*det):
                return False

        kpt = cv2.KeyPoint()
        kpt.pt = ((c + xc) * (2**octave_index), (r + xr) * (2**octave_index))
        kpt.octave = octave_index + image_index*(2**8) + int(np.round((xi + 0.5)*255))*(2**16)
        kpt.size = sigma * pow(2, (image_index + xi) / n_octave_layers)*(2**octave_index)*2        
        kpt.response = abs(contr)

        return kpt, r, c, image_index


    @staticmethod
    def calc_orientation_hist(img, kpt, octave_index, radius, sigma, n):
        # print("Calculating orientation hist...")
        img = np.array(img, dtype=np.float32)
        hist = np.zeros(n, dtype=np.float32)
        temphist = np.zeros(n, dtype=np.float32)
        length = (radius*2+1)**2
        expf_scale = -1 / (2 * (sigma**2))

        X = np.zeros(length)
        Y = np.zeros(length)
        W = np.zeros(length)

        k = 0
        for i in range(-radius, radius+1):
            # y = int(pt[1] + i)
            y = int(np.round(kpt.pt[1] / np.float32(2 ** octave_index))) + i
            if (y <= 0 or y >= img.shape[0] - 1):
                continue
            for j in range(-radius, radius+1):
                # x = int(pt[0] + j)
                x = int(np.round(kpt.pt[0] / np.float32(2 ** octave_index))) + j
                if (x<=0 or x >= img.shape[1] - 1):
                    continue
                
                dx = img[y, x+1] - img[y, x-1]
                dy = img[y-1, x] - img[y+1, x]
                X[k], Y[k], W[k] = dx, dy, (i**2 + j**2)*expf_scale
                k += 1

        length = k

        # compute gradient values, orientations and the weights over the pixel neighborhood
        W = np.exp(W[:length])
        Ori = np.rad2deg(np.arctan2(Y[:length], X[:length]))
        Mag = np.sqrt(np.power(X[:length], 2) + np.power(Y[:length], 2))

        for k in range(length):
            bin_vote = np.round((n/360)*Ori[k]).astype(int) % n
            temphist[bin_vote] += W[k]*Mag[k]

        # smooth the histogram
        
        for i in range(n):
            hist[i] = (temphist[i-2] + temphist[(i+2)%n])*(1/16) + (temphist[i-1] + temphist[(i+1)%n])*(4/16) + temphist[i]*(6/16)

        maxval = np.max(hist)

        return maxval, hist

    @staticmethod
    def compare_keypoints(keypoint1, keypoint2):
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    @staticmethod
    def remove_duplicate_keypoints(keypoints, n_points):
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(SIFT.compare_keypoints))
        unique_keypoints = [keypoints[0]]


        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        
        if len(unique_keypoints) > n_points:
            threshold = - np.partition(np.array([-kpt.response for kpt in unique_keypoints]), n_points)[n_points-1]
            return [keypoint for keypoint in unique_keypoints if keypoint.response >= threshold]
        else:
            return unique_keypoints
    

    @staticmethod
    def convert_keypoints_to_input_image_size(keypoints):
        """Convert keypoint point, size, and octave to input image size
        """
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    @staticmethod
    def calc_descriptors(pyr, keypoints, n_octave_layers):
        d, n = SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS

        descriptors = []
        for kpt in tqdm(keypoints):
            octave = kpt.octave & 255
            layer = (kpt.octave >> 8) & 255
            if octave >= 128:
                octave = octave | -128
            if octave >= 0:
                scale = 1 / (1 << octave) 
            else: 
                scale = 1 << -octave
            # assert (octave >= -1 and layer <= n_octave_layers+2)
            size = kpt.size * scale
            ptf = (kpt.pt[0]*scale, kpt.pt[1]*scale)
            img = pyr[(octave + 1),  layer]
            angle = 360 - kpt.angle
            
            if np.zeros_like(abs(angle - 360)):
                angle = 0
            # print(f"Calculating descriptors at ({kpt.pt[0]}, {kpt.pt[1]})")
            descriptor = SIFT.calc_SIFT_descriptor(img, ptf, angle, size*0.5, d, n)
            descriptors.append(descriptor)
        
        return descriptors

    @staticmethod
    def calc_SIFT_descriptor(img, ptf, ori, scl, d, n):
        img = img.astype(np.float32)
        rows, cols = img.shape
        pt = np.round(np.array(ptf)).astype(int)
        cos_t, sin_t = math.cos(np.deg2rad(ori)), math.sin(np.deg2rad(ori))
        bins_per_rad = n / 360
        exp_scale = 1 / ((d**2)*0.5)
        hist_width = SIFT_DESCR_SCL_FCTR * scl
        
        radius = int(np.round(hist_width * math.sqrt(2) * (d+1) * 0.5))
        radius = int(min(radius, math.sqrt(rows ** 2 + cols ** 2)))
        
        cos_t /= hist_width
        sin_t /= hist_width

        length = (radius*2+1)**2
        hist_length = ((d+2)**2) * (n+2)

        X = np.zeros(length, dtype=np.float32)
        Y = np.zeros(length, dtype=np.float32)
        Mag = np.zeros(length, dtype=np.float32)
        W = np.zeros(length, dtype=np.float32)
        RBin = np.zeros(length, dtype=np.float32)
        CBin = np.zeros(length, dtype=np.float32)
        hist = np.zeros(hist_length, dtype=np.float32)
        
        k = 0
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                c_rot, r_rot = j * cos_t - i *sin_t,  j * sin_t + i * cos_t
                rbin, cbin = r_rot + d/2 - 0.5, c_rot + d/2 - 0.5
                r, c = pt[1]+i , pt[0]+j

                if (rbin > -1 and rbin < d and cbin > -1 and cbin < d and
                    r > 0 and r < rows - 1 and c > 0 and c < cols - 1):
                    dx = img[r, c+1] - img[r, c-1]
                    dy = img[r-1, c] - img[r+1, c]
                    X[k], Y[k], RBin[k], CBin[k] = dx, dy, rbin, cbin
                    W[k] = (c_rot**2 + r_rot**2)*exp_scale
                    k += 1

        length = k
        W = np.exp(W[:length])
        Ori = np.rad2deg(np.arctan2(Y[:length], X[:length]))
        Mag = np.sqrt(np.power(X[:length], 2) + np.power(Y[:length], 2))

        for k in range(length):
            rbin, cbin = RBin[k], CBin[k]
            obin = (Ori[k] - ori) * bins_per_rad
            mag = Mag[k] * W[k]

            r0, c0, o0 = tuple(np.floor((rbin, cbin, obin)).astype(int))
            rbin -= r0
            cbin -= c0
            obin -= o0
            
            o0 =  o0 % n

            v_r1 = mag*rbin
            v_r0 = mag - v_r1
            v_rc11 = v_r1*cbin
            v_rc10 = v_r1 - v_rc11
            v_rc01 = v_r0*cbin
            v_rc00 = v_r0 - v_rc01
            v_rco111 = v_rc11*obin
            v_rco110 = v_rc11 - v_rco111
            v_rco101 = v_rc10*obin
            v_rco100 = v_rc10 - v_rco101
            v_rco011 = v_rc01*obin
            v_rco010 = v_rc01 - v_rco011
            v_rco001 = v_rc00*obin
            v_rco000 = v_rc00 - v_rco001

            # print(r0, c0, o0)
            idx = ((r0+1)*(d+2) + c0+1)*(d+2) + o0
            # print(idx)
            hist[idx] += v_rco000
            hist[idx+1] += v_rco001
            hist[idx+(n+2)] += v_rco010
            hist[idx+(n+3)] += v_rco011
            hist[idx+(d+2)*(n+2)] += v_rco100
            hist[idx+(d+2)*(n+2)+1] += v_rco101
            hist[idx+(d+3)*(n+2)] += v_rco110
            hist[idx+(d+3)*(n+2)+1] += v_rco111
        
        length = d*d*n
        dst = np.zeros(length, dtype=np.float32)
        for i in range(d):
            for j in range(d):
                idx = ((i+1)*(d+2) + (j+1))*(n+2)
                hist[idx] += hist[idx+n]
                hist[idx+1] += hist[idx+n+1]
                for k in range(n):
                    dst[(i*d + j)*n + k] = hist[idx+k]
                    
        thr = norm(dst)*SIFT_DESCR_MAG_THR
        dst = np.minimum(dst, thr)
        nrm2 = SIFT_INT_DESCR_FCTR/max(norm(dst), sys.float_info.epsilon)
        dst = np.maximum(np.minimum(np.round(dst*nrm2), 255), 0)
        return dst

    @staticmethod
    def detect_and_compute(img):
        base_img = SIFT.create_initial_image(img, sigma=1.6)
        pyr = SIFT.build_gaussian_pyramid(base_img, sigma=1.6, n_octave_layers=3)
        dog_pyr = SIFT.build_DoG_pyramid(pyr)
        keypoints = SIFT.find_scale_space_extrema(pyr, dog_pyr,  n_octave_layers=3, sigma=1.6, image_border_width=5, contrast_threshold=0.06)
        
        keypoints = SIFT.remove_duplicate_keypoints(keypoints, n_points=1000)
        keypoints = SIFT.convert_keypoints_to_input_image_size(keypoints)
        descriptors = SIFT.calc_descriptors(pyr, keypoints, n_octave_layers=3)

        return keypoints, descriptors

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test set with pretrained inceptionv3 model.')
    parser.add_argument('--input', '-i', type=str, default=None, help='Input folder path for all images', required=True)
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for keypoint, descriptors npy', required=True)
    return parser.parse_args()

def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Make Directory at path {directory}")


if __name__ == '__main__':
    args = get_args()
    
    filenames = sorted(os.listdir(args.input))[4:]
    print(filenames)
    safe_mkdir(args.output)

    sift = SIFT()
    
    for filename in filenames:
        img = cv2.imread(os.path.join(args.input, filename), cv2.IMREAD_COLOR)
        
        scale_percent = 10 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        
        keypoints, descriptors = sift.detect_and_compute(img)
        keypoints_to_sava = [(point.pt, point.size, point.angle, point.response, point.octave, 
        point.class_id) for point in keypoints]
        np.save(os.path.join(args.output, f"{filename[:-4]}_keypoint.npy"), keypoints_to_sava)
        np.save(os.path.join(args.output, f"{filename[:-4]}_descriptor.npy"), descriptors)
        gray = cv2.drawKeypoints(gray, keypoints, gray)
        plt.imsave(os.path.join(args.output, f"{filename[:-4]}_keypoiny.png"), gray)
        