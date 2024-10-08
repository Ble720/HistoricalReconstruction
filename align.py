import cv2
import numpy as np
import os

MATCH_THRESH = 0.75

def align_perspective(img_query_path, img_reference_path, save_dir):
    
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()

    img_query = cv2.imread(img_query_path)
    img_reference = cv2.imread(img_reference_path)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    img_reference = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
 

    kp1, des1 = orb.detectAndCompute(img_query,None)
    kp2, des2 = orb.detectAndCompute(img_reference,None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)

    '''
    src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    ## find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = img_query.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    ## draw found regions
    img2 = cv2.polylines(img_reference, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("found", img_reference)

    ## draw match lines
    res = cv2.drawMatches(img_query, kp1, img_reference, kp2, matches[:20],None,flags=2)

    cv2.imshow("orb_match", res)

    cv2.waitKey();cv2.destroyAllWindows()


    '''
    numGoodMatches = int(len(matches) * MATCH_THRESH)
    matches = matches[:numGoodMatches]

    src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    h, status = cv2.findHomography(src_pts, dst_pts)

    height, width = img_query.shape
    img_query_fix = cv2.warpPerspective(img_query, h, (width, height))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(save_dir + '/align.jpg', img_query_fix)

def align_color(img_query_path, img_reference_path, save_dir):
    img_query = cv2.imread(img_query_path)
    img_reference = cv2.imread(img_reference_path)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    img_reference = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    
    query_contrast = img_query.std()
    reference_contrast = img_reference.std()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(save_dir + '/align_color.jpg', img_query)

def align_histogram(img_query_path, img_reference_path, save_dir):
    img_query = cv2.imread(img_query_path)
    img_reference = cv2.imread(img_reference_path)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    img_reference = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img_clache = clahe.apply(img_query)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(save_dir + '/align_clache_q.jpg', img_clache)
    return img_clache

def save_im(img, name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(save_dir + '/' + name + '.jpg', img)


if __name__ == '__main__':
    src_dir = './image_pair'
    save_dir = './gen'
    num_pairs = len(os.listdir(src_dir))
    for i in range(num_pairs):
        src_path = src_dir + '/' + str(i+1)
        pair_names = os.listdir(src_path)
        if pair_names[0][:3] == 'SRC':
            query = src_path + '/' + pair_names[0]
            reference = src_path + '/' + pair_names[1]
        else:
            query = src_path + '/' + pair_names[1]
            reference = src_path + '/' + pair_names[0]

        align_histogram(query, reference, save_dir + '/' + str(i+1))