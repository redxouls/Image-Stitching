import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
import random
from matplotlib import pyplot as plt
import glob
from PIL import Image
from SIFT import SIFT
from tqdm import tqdm
import os


filenames = sorted(os.listdir("./parrington"), reverse=True)
filenames = [filename for filename in filenames if filename.startswith("prtn")]
print(filenames)

imgs = []
keypoints, descriptors = [], []
scale_percent = 50 # percent of original size

for filename in filenames[16:18]:
    img = cv.imread(f"./parrington/{filename}")
    if scale_percent != 100:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    imgs.append(img)

    sift = SIFT()
    k, d = sift.detect_and_compute(img)
    
    # k_raw = np.load(f"./data/{filename[:-4]}_keypoint.npy", allow_pickle=True)
    # k = [cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5]) for point in k_raw]
    # d = np.load(f"./data/{filename[:-4]}_descriptor.npy", allow_pickle=True)

    keypoints.append(k)
    descriptors.append(d)
    print(f"Filename: {filename} , Keypoints#: {len(k)}")

imgs = np.array(imgs)


def match_feat(kp, des):
  tree = [KDTree(des[0]), KDTree(des[1])]
  match = [[], []]
  
  for point in des[1]:
    dist0, ind0 = tree[0].query(point, k=1)
    dist1, ind1 = tree[1].query(des[0][ind0], k=1)
    if des[1][ind1][0] == point[0] and des[1][ind1][1] == point[1]:
      match[0].append(kp[0][ind0].pt)
      match[1].append(kp[1][ind1].pt)
  print(len(match[1]))
  # for i in (0,1):
  #   cv.drawKeypoints(img[i], match[i], img[i])
  #   cv.imwrite('sift_keypoints'+ str(i) +'.jpg', img[i])
  return np.array(match[0]), np.array(match[1])

p1, p2 = match_feat(keypoints[:2], descriptors[:2])


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

total_img = np.concatenate((imgs[0], imgs[1]), axis=1)
# plot_matches(p1, p2, total_img)


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

def ransac(p1, p2, thresh = 10, k = 4, iter = 2000):
  max_match = 0
  all_match = []
  best_match_pairs = [[], []]
  best_H = None
  for i in range(iter):
    match = 0
    match_pairs = [[],[]]
    sam = random.sample( range(p2.shape[0]), k )
    H = homography( p1[sam], p2[sam] )

    # check rank 
    if np.linalg.matrix_rank(H) < 3:
      continue
    for i in range(p2.shape[0]):
      p = np.append(p2[i], 1)
      pred = np.dot(H, p)
      pred = np.array( [ pred[0]/pred[2], pred[1]/pred[2] ])
      if np.linalg.norm(pred - p1[i], ord = 2) < thresh:
        match += 1
        match_pairs[0].append(p1[i])
        match_pairs[1].append(p2[i])
    all_match.append(match)
    if match > max_match:
      max_match = match
      best_match_pairs = match_pairs
      best_H = H.copy()
    
  print("Inliers / Total pairs:", max_match, "/", p2.shape[0], "(", max_match/p2.shape[0], ")")
  return best_H, np.array(best_match_pairs)

H, pairs = ransac(p1, p2)
total_img = np.concatenate((imgs[0], imgs[1]), axis=1)
plot_matches(pairs[0], pairs[1], total_img)


def stitch_img(left, right, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


plt.imshow(stitch_img(imgs[1], imgs[0], H))
plt.show()
