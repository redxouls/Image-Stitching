import os
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from PIL import Image
from tqdm import tqdm
from utils import Partial_img, SIFT, cylindrical_warp, match_feat, homography, cylin_affine, ransac, stitch_img, get_args, safe_mkdir

if __name__ == '__main__':
	args = get_args()
	safe_mkdir(args.output)
	filenames = sorted(glob.glob(os.path.join(args.input, f"*.JPG")))
	print(f"Located filenames: {str(filenames)}")
	
	img = []
	kp, des = [], []
	pre, cur = None, None

	img = cv.imread(filenames[0])
	resize_ratio = 8
	dim = ( img.shape[0] // resize_ratio, img.shape[1] // resize_ratio )
	print("Resized dimension:", dim)

	# assume we can get the focal length estimate
	focal = []
	with open(os.path.join(args.input, "pano.txt")) as f:
		ind = 1
		for line in f:
			if ind % 13 == 12:
				focal.append(float(line))
			ind += 1
	print(f"Focal lengths: {str(focal)}")

	panorama = None
	for i in tqdm(range(len(filenames))):
		print(f"Processing: {str(filenames[i])}")
		img = cv.resize(cv.imread(filenames[i]), (dim[1], dim[0]))
		img = cylindrical_warp(img, focal[i])
		k, d = SIFT(img)

		cur = Partial_img(img, k, d)
		print( filenames[i], "keypoint num:", len(k))
		if i == 0:
			panorama = cur.img
			pre, cur = cur, None
			continue
		p_pre, p_cur = match_feat(pre.kp, pre.des, cur.kp, cur.des)
		A, pairs = ransac(p_pre, p_cur)

		panorama, cur_img = stitch_img(panorama, cur.img, A)
		cur.img = cur_img
		cur.kp, cur.des = SIFT(cur_img)

		cv.imwrite(os.path.join(args.output, f'panorama_{i}.jpg'), panorama)
		
		pre, cur = cur, None

