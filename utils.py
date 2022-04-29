import os
import argparse
from time import time
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
import random
from matplotlib import pyplot as plt
import glob
from PIL import Image
from tqdm import tqdm
from SIFT import SIFT as S


class Partial_img:
	def __init__(self, img = None, kp = None, des = None):
		self.img = img
		self.kp = kp
		self.des = des

def SIFT(img):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	sift = S()
	k, d = sift.detect_and_compute(gray)
	d = np.array(d)
	return k, d

def cylindrical_warp(src, f):
	print("Starting cylindrical warping...")
	map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
	map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
	
	c = [ src.shape[0] / 2.0, src.shape[1] / 2.0 ]
	f2 = f ** 2
	
	
	for i in range(src.shape[0]):
		map_x[i] = (f * np.tan( 1.0 * (np.arange(src.shape[1]) - c[1]) / f ) + c[1])
		map_y[i] = 1.0 * (i - c[0]) / f * np.sqrt( (map_x[i] - c[1])**2 + f2 ) + c[0]
		
	warp = cv.remap(src, map_x, map_y, cv.INTER_LINEAR,	borderMode = cv.BORDER_CONSTANT)
	cutoff = [0, src.shape[1] - 1]
	black = np.zeros(3)  # Black pixel.
	while np.array_equal(warp[ int(warp.shape[0]/2), cutoff[0] ], black):
		cutoff[0] += 1
	while np.array_equal(warp[ int(warp.shape[0]/2), cutoff[1] ], black):
		cutoff[1] -= 1
	warp = warp[ :, cutoff[0]+2 : cutoff[1]-1 ]
	
	return warp

def match_feat(kp_tar, des_tar, kp_src, des_src):
	print("Start matching feature...")
	tree = [KDTree(des_tar), KDTree(des_src)]
	match = [[], []]
	
	for point in des_src:
		dist0, ind0 = tree[0].query(point, k=2)
		if dist0[0] / dist0[1] > 0.5:
			continue
		dist0, ind0 = dist0[0], ind0[0]
		dist1, ind1 = tree[1].query(des_tar[ind0], k=1)
		if des_src[ind1][0] == point[0] and des_src[ind1][1] == point[1]:
			match[0].append( kp_tar[ind0].pt )
			match[1].append( kp_src[ind1].pt )
	return np.array(match[0]), np.array(match[1])



def plot_matches(p1, p2, total_img):
	match_img = total_img.copy()
	offset = total_img.shape[1]/2
	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
	
	ax.plot(p1[:, 0], p1[:, 1], 'xr')
	ax.plot(p2[:, 0] + offset, p2[:, 1], 'xr')
	 
	ax.plot([p1[:, 0], p2[:, 0] + offset], [p1[:, 1], p2[:, 1]],
			'r', linewidth=0.1)

	plt.show()

def homography(p1, p2):
	rows = []
	for a, b in zip(p1, p2):
		row1 = [0, 0, 0, b[0], b[1], 1, -a[1]*b[0], -a[1]*b[1], -a[1]]
		row2 = [b[0], b[1], 1, 0, 0, 0, -a[0]*b[0], -a[0]*b[1], -a[0]]
		rows.append(row1)
		rows.append(row2)
	rows = np.array(rows)
	U, s, V = np.linalg.svd(rows)
	H = V[-1].reshape(3, 3)
	H = H/H[2, 2] # standardize to let w*H[2,2] = 1
	return H

def cylin_affine(p1, p2):
	A = np.zeros( (p1.shape[0] * 2, 6) )
	B = np.zeros( (p1.shape[0] * 2, 1) )
	ind = 0
	A_rows = []
	B_rows = []
	for a, b in zip(p1, p2):
		row1 = [1, 0, b[0], b[1], 0, 0]
		row2 = [0, 1, 0, 0, b[0], b[1]]
		A_rows.append(row1)
		A_rows.append(row2)
		B_rows.append( [a[0]] )
		B_rows.append( [a[1]] )
	A = np.array(A_rows)
	B = np.array(B_rows)
	cyl_aff , _, _, _ = np.linalg.lstsq(A, B, rcond=-1)
	cyl_aff = np.array( [[ cyl_aff[2], cyl_aff[3], cyl_aff[0] ], [ cyl_aff[4], cyl_aff[5], cyl_aff[1]]] ).squeeze()
	return cyl_aff

def ransac(tar, src, thresh = 0.5, k = 3, iter = 5000, method = "affine"):
	print("Start Ransac ...")
	max_match = 0
	all_match = []
	best_match_pairs = [[], []]
	best_trans = None
	for i in range(iter):
		match = 0
		match_pairs = [[],[]]
		sam = random.sample( range(src.shape[0]), k )

		if method == "homography":
			H = homography( tar[sam], src[sam] )
		elif method == "affine":
			A = cylin_affine( tar[sam], src[sam] )

		# # check rank 
		# if np.linalg.matrix_rank(H) < 3:
		#   continue
		for i in range(src.shape[0]):
			p = np.append(src[i], 1)
			if method == "homography":
				pred = np.dot(H, p)
				pred = np.array( [ pred[0]/pred[2], pred[1]/pred[2] ])
			elif method == "affine":
				pred = np.dot(A, p)
			if np.linalg.norm(pred - tar[i], ord = 2) < thresh:
				match += 1
				match_pairs[0].append(tar[i])
				match_pairs[1].append(src[i])
			all_match.append(match)
			if match > max_match:
				max_match = match
				best_match_pairs = match_pairs
				if method == "homography":
					best_trans = H.copy()
				elif method == "affine":
					best_trans = A.copy()
		
	print("Inliers / Total pairs:", max_match, "/", src.shape[0], "(", max_match/src.shape[0], ")")
	return best_trans, np.array(best_match_pairs)


def stitch_img(img1, img2, trans):
	print("Stitching image")
	# stitch img2 to img1
	corners = np.array([[0, 0, 1], [0, img2.shape[0], 1], [img2.shape[1], 0, 1], [img2.shape[1], img2.shape[0], 1]]).T
	
	# find corners of the warped image
	mapped_corners = np.dot(trans, corners).T.astype("int32")

	# shift the warpped image downwards if the transformed y coordinate is a negative value
	min_v = min(0, min(mapped_corners[:, 1]))
	max_v = max(img1.shape[0], max(mapped_corners[:, 1]))
	max_h = max(img1.shape[1], max(mapped_corners[:, 0]))
	
	mapped_h = [ min(mapped_corners[:, 0]), max(mapped_corners[:, 0]) ]
	mapped_v = [ min(mapped_corners[:, 1]) - min_v, max(mapped_corners[:, 1]) - min_v ]

	trans_new = trans.copy()
	trans_new[1, 2] += abs(min_v)
	
	# shift the base image downwards if the transformed y coordinate is a negative value
	top_pad = np.zeros( ( abs(min_v), img1.shape[1], 3 ) )
	bot_pad = np.zeros( ( max_v - img1.shape[0], img1.shape[1], 3 ) )
	right_pad = np.zeros( ( max_v - min_v, max_h - img1.shape[1], 3) )

	warped_l = np.concatenate( (top_pad, img1, bot_pad), axis=0 )
	warped_l = np.concatenate( (warped_l, right_pad), axis=1 ).astype('uint32')
	
	w, h = max_h, max_v - min_v
	warped_r = cv.warpAffine(src = img2, M = trans_new, dsize = (w, h)).astype('uint32')

	weight = np.zeros( (h, mapped_h[1]) )
	black = np.zeros(3)  # Black pixel.

	# clean up horizontal edges from affine transform
	for j in range(mapped_h[0], img1.shape[1]):
		u, d = mapped_v[0], mapped_v[1]-1
		while u < mapped_v[1]-4 and np.array_equal(warped_r[ u, j ], black):
			u += 1
		while d > mapped_v[0]+3 and np.array_equal(warped_r[ d, j ], black):
			d -= 1
		for offset in range(3):
			warped_r[u + offset, j] = black
			warped_r[d - offset, j] = black

	# clean up vertical edges from affine transform and define weight (for blending)
	for i in range( mapped_v[0], mapped_v[1] ):
		l, r = mapped_h[0], mapped_h[1]-1
		while l < mapped_h[1]-4 and np.array_equal(warped_r[ i, l ], black):
			l += 1
		while r > mapped_h[0]+3 and np.array_equal(warped_r[ i, r ], black):
			r -= 1
		for offset in range(3):
			warped_r[i, l + offset] = black
			warped_r[i, r - offset] = black
		
		img1_r = img1.shape[1] - 1
		while img1_r > l+4 and np.array_equal(warped_l[ i, img1_r ], black):
			img1_r -= 1

		if img1_r - (l+4) < 0:
			continue
		interval = 1.0 / (img1_r - (l+4) + 2)
		ind = 1
		for j in range(l+4, img1_r+1):
			weight[i, j] = ind * interval
			ind += 1

	# Stitching procedure, store results in warped_l.
	for i in range( mapped_v[0], mapped_v[1] ):
		for j in range( mapped_h[0], mapped_h[1] ):
			pixel_l = warped_l[i, j, :]
			pixel_r = warped_r[i, j, :]
			
			if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
				warped_l[i, j, :] = pixel_l
			elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
				warped_l[i, j, :] = pixel_r
			elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
				warped_l[i, j, :] = pixel_l * (1-weight[i,j]) + pixel_r * weight[i,j]
			else:
				pass
					
	stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :].astype("uint8")
	return stitch_image, warped_r.astype("uint8")

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test set with pretrained inceptionv3 model.')
    parser.add_argument('--input', '-i', type=str, default=None, help='Input folder path for all images', required=True)
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for keypoint, descriptors npy', required=True)
    return parser.parse_args()

def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Make Directory at path {directory}")
