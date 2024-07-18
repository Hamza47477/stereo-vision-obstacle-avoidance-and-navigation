import numpy as np
import cv2
import glob
import pickle
import os

class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(os.path.join(cal_path, 'right', '*.png'))
        images_left = glob.glob(os.path.join(cal_path, 'left', '*.png'))
        images_left.sort()
        images_right.sort()

        img_shape = None  # Initialize img_shape to None

        if len(images_left) != len(images_right):
            raise ValueError("Number of left and right images do not match")

        for i, fname in enumerate(images_right):
            print(f'Processing pair {i+1}: {images_left[i]} and {images_right[i]}')
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            if img_l is None or img_r is None:
                print(f'Error loading images: {images_left[i]} or {images_right[i]}')
                continue

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret_l and ret_r:
                print(f'Chessboard corners found in pair {i+1}')
                self.objpoints.append(self.objp)

                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and save the corners
                img_l = cv2.drawChessboardCorners(img_l, (9, 6), corners_l, ret_l)
                cv2.imwrite(os.path.join(cal_path, 'calibration_results', f'left_{i+1}.jpg'), img_l)
                cv2.imshow('Left Image', img_l)
                cv2.waitKey(500)

                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and save the corners
                img_r = cv2.drawChessboardCorners(img_r, (9, 6), corners_r, ret_r)
                cv2.imwrite(os.path.join(cal_path, 'calibration_results', f'right_{i+1}.jpg'), img_r)
                cv2.imshow('Right Image', img_r)
                cv2.waitKey(500)

                if img_shape is None:
                    img_shape = gray_l.shape[::-1]
            else:
                print(f'Chessboard corners not found in pair {i+1}')

        if img_shape is None:
            raise ValueError("No chessboard corners found in images.")

        print("Calibrating cameras...")
        ret, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        print("Left camera calibration complete")
        ret, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)
        print("Right camera calibration complete")

        self.camera_model = self.stereo_calibrate(img_shape)

        # Save the calibration parameters
        with open(os.path.join(cal_path, 'calibration_results', 'calibration_data.pkl'), 'wb') as f:
            pickle.dump(self.camera_model, f)
        print("Calibration parameters saved.")

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        # Calculate rectification parameters
        R1, R2, P1, P2, Q, validROIL, validROIR = cv2.stereoRectify(M1, d1, M2, d2, dims, R, T)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        print('R1', R1)
        print('R2', R2)
        print('P1', P1)
        print('P2', P2)
        print('Q', Q)
        print('')

        # Compute undistortion and rectification transformation maps
        undistL, rectifL = cv2.initUndistortRectifyMap(M1, d1, R1, P1, dims, cv2.CV_32FC1)
        undistR, rectifR = cv2.initUndistortRectifyMap(M2, d2, R2, P2, dims, cv2.CV_32FC1)
        
        

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                        ('dist2', d2), ('rvecs1', self.r1),
                        ('rvecs2', self.r2), ('R', R), ('T', T),
                        ('E', E), ('F', F), ('R1', R1), ('R2', R2),
                        ('P1', P1), ('P2', P2), ('Q', Q)])

        # Save the calibration parameters to text files
        output_dir = os.path.join(self.cal_path, 'calibration_results')
        np.savetxt(os.path.join(output_dir, 'Q.txt'), Q, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'FundMat.txt'), F, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'CmL.txt'), M1, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'CmR.txt'), M2, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'DcL.txt'), d1, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'DcR.txt'), d2, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'Rtn.txt'), R, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'Trnsl.txt'), T, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'RectifL.txt'), R1, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'RectifR.txt'), R2, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'ProjL.txt'), P1, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'ProjR.txt'), P2, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'umapL.txt'), undistL, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'rmapL.txt'), rectifL, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'umapR.txt'), undistR, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'rmapR.txt'), rectifR, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'ROIL.txt'), validROIL, fmt='%.5e')
        np.savetxt(os.path.join(output_dir, 'ROIR.txt'), validROIR, fmt='%.5e')

        print(f'Parameters saved to folder: [{output_dir}]')

        cv2.destroyAllWindows()
        return camera_model


if __name__ == '__main__':
    # Specify the path to the folder containing the images
    filepath = r'images copy'
    
    # Create a directory for the calibration results if it doesn't exist
    if not os.path.exists(os.path.join(filepath, 'calibration_results')):
        os.makedirs(os.path.join(filepath, 'calibration_results'))
    
    cal_data = StereoCalibration(filepath)
