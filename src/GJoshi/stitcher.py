import pdb
import glob
import cv2
import os
import numpy as np
import time

class PanaromaStitcher():
    def __init__(self):
        # Initializing the SIFT detector
        self.sift = cv2.SIFT_create()
        # BFMatcher with KNN for better filtering of matches
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def make_panaroma_for_images_in(self, path, no_images = 5, lowe = 0.75):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')

        if len(all_images) < 2:
            raise ValueError("At least two images are required to stitch a panorama.")
        
        images = [cv2.imread(img_path) for img_path in all_images]
        scaling_factor = 0.5
        images_resized = [cv2.resize(img, (int(img.shape[1] * scaling_factor), 
                                           int(img.shape[0] * scaling_factor))) for img in images]
        
        if len(images_resized) == 5:
            Order = [2,3,4,1,0]
            no_images = 5
        elif len(images_resized) == 6:
            Order = [3,2,4,1,5,0]
            no_images = 5

        start_time = time.time()

        i = 0
        panorama = images_resized[Order[i]]
        homography_matrix_list = []
        i = i + 1
        no_images_used = 1

        for j in range(no_images -1):
            # print(i)
            plane_image = panorama; project_image = images_resized[Order[i]]

            panorama, H, sucess = self.project_img(plane_image, project_image, lowe = 0.75)
            if sucess == 1:
                no_images_used = no_images_used +1
            
            homography_matrix_list.append(H)
            i = i+1
  
        print("No of images used: ",no_images_used)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed time: ",elapsed_time) 

        return panorama, homography_matrix_list
    
    def project_img(self, plane_image, project_image, lowe = 0.6):
        # Initialize SIFT detector and BFMatcher
        sift = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        # Detect keypoints and descriptors for the first two images
        kp1, des1 = sift.detectAndCompute(plane_image, None)
        kp2, des2 = sift.detectAndCompute(project_image, None)


        # Match descriptors using KNN and apply Lowe's ratio test
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < lowe * n.distance]

        matched_img = cv2.drawMatches(plane_image, kp1, project_image, kp2, good_matches, 
                                None, matchColor = (0, 255, 0), singlePointColor = (255, 0, 0),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        panorama = plane_image; H = 0; sucess = 0
        
        if len(good_matches) >= 50:  # increased threshold for better matches
            # Extract matching keypoints positions

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # H, mask = cv2.findHomography(dst_pts,src_pts,  cv2.RANSAC, 5.0) # mask is useless

            H, mask = self.ransac_homography(dst_pts, src_pts) 
            # print("H", H)
            # print("H2", H2)

            h1, w1 = plane_image.shape[:2]
            h2, w2 = project_image.shape[:2]

            # Get points for the second image corners and transform them
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners_img2, H)
            
            # Combine all points to determine bounding box for the panorama
            all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), transformed_corners), axis=0)
            # print("all_corners: ", all_corners.shape , all_corners)

            # Get minimum and maximum coordinates
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            # Translation distance for positive coordinates
            translation_dist = [-xmin, -ymin]
            
            # Adjust homography to include translation
            H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

            panorama = cv2.warpPerspective(project_image, H_translation @ H, (xmax - xmin, ymax - ymin))
            # panorama = warp_perspective_manual(project_image, H_translation @ H, (xmax - xmin, ymax - ymin))
            # print(panorama.shape)

            # Ensure plane_image fits in panorama
            if (h1, w1) == plane_image.shape[:2] :
                y_start = translation_dist[1]
                x_start = translation_dist[0]

                mask = (plane_image != [0, 0, 0]).any(axis=2)

                # print("dest: ",panorama[y_start:y_start + h1, x_start:x_start + w1].shape)
                # print("h1, w1: ", h1, w1)
                # print("mask:", mask.shape)
                # print(plane_image.shape)

                    # Use numpy.where to blend images based on the mask
                panorama[y_start:y_start + h1, x_start:x_start + w1] = np.where(
                    mask[..., None], plane_image, panorama[y_start:y_start + h1, x_start:x_start + w1])

            else:
                print("Sizes are not same")
            
            # return panorama, H
            sucess = sucess + 1
        else:
            print("Not enough good matches to create a panorama.")   

        return panorama, H, sucess

    def compute_homography(self, src_pts, dst_pts):
        """
        Computes the homography matrix from source points to destination points using DLT.
        Arguments:
            src_pts: Source points, shape (N, 2)
            dst_pts: Destination points, shape (N, 2)
        Returns:
            Homography matrix H (3x3)
        """

        num_points = src_pts.shape[0]
        A = []

        for i in range(num_points):
            x_src, y_src = src_pts[i][0], src_pts[i][1]
            x_dst, y_dst = dst_pts[i][0], dst_pts[i][1]

            A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst])
            A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_dst, y_src * y_dst, y_dst])

        A = np.array(A)

        # Use SVD to solve for H (or smallest eigen vector A.T A)
        # U, S, V = np.linalg.svd(A)
        # U, S, V = np.linalg.svd(A.T@A)
        U, S, V = np.linalg.svd(A)
        H = V[-1, :].reshape((3, 3))

        # Normalize so that H[2,2] is 1
        H /= H[2, 2]
        return H
    
    def ransac_homography(self, src_pts, dst_pts, threshold=5.0, max_iterations=1000):
        """
        RANSAC to robustly estimate the homography matrix
        Arguments:
            src_pts: Source points, shape (N, 2)
            dst_pts: Destination points, shape (N, 2)
            threshold: Distance threshold for inliers
            max_iterations: Maximum number of RANSAC iterations
        Returns:
            Best homography matrix H and inlier mask
        """

        src_pts = src_pts.reshape(-1, 2)  # Reshape to (len(), 2)
        dst_pts = dst_pts.reshape(-1, 2)
                            
        best_H = None
        max_inliers = 0
        best_inliers_mask = None

        num_points = src_pts.shape[0]
        
        # H = compute_homography(src_pts, dst_pts)
        # return H, 0

        for _ in range(max_iterations):
            # Randomly select 4 points to compute homography
            indices = np.random.choice(num_points, 4, replace=False)
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]

            # Compute homography from these 4 points
            H = self.compute_homography(src_sample, dst_sample)

            # Project src_pts using H to find inliers
            projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            distances = np.linalg.norm(projected_pts - dst_pts, axis=1)

            # Determine inliers
            inliers_mask = distances < threshold
            num_inliers = np.sum(inliers_mask)

            # Check if this model is the best so far
            if num_inliers > max_inliers:
                best_H = H
                max_inliers = num_inliers
                best_inliers_mask = inliers_mask

        return best_H, best_inliers_mask

    def warp_and_stitch(self, img1, img2, H):
        # Compute size of output panorama
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Get corners of img2, apply homography
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_img2, H)
        all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), transformed_corners), axis=0)

        # Find the bounds of the stitched image
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_dist = [-xmin, -ymin]

        # Adjust homography to include translation
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        output_img = cv2.warpPerspective(img2, H_translation @ H, (xmax - xmin, ymax - ymin))
        output_img[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = img1

        return output_img

