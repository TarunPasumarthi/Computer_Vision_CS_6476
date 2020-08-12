import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time


def objective_func(x, **kwargs):
    """
    Calculates the difference in image (pixel coordinates) and returns 
    it as a 2*n_points vector

    Args: 
    -        x: numpy array of 11 parameters of P in vector form 
                (remember you will have to fix P_34=1) to estimate the reprojection error
    - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                retrieve these 2D and 3D points and then use them to compute 
                the reprojection error.
    Returns:
    -     diff: A N_points-d vector (1-D numpy array) of differences betwen 
                projected and actual 2D points

    """

    diff = None

    ##############################
    # TODO: Student code goes here
    P=np.array([[x[0],x[1],x[2],x[3]], [x[4],x[5],x[6],x[7]], [x[8],x[9],x[10],1]])
    x_3d= kwargs["pts3d"]
    x_2d= kwargs["pts2d"]
    xp= projection(P,x_3d)
    diff= xp-x_2d
    diff=diff.flatten()

    # raise NotImplementedError
    ##############################

    return diff


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    projected_points_2d = None

    ##############################
    # TODO: Student code goes here
    if(points_3d.shape[1]==3):
        n_ones= np.ones((points_3d.shape[0],1))
        points_3d= np.hstack((points_3d,n_ones))
        
    projected_homo= np.dot(P,points_3d.T)
    x=projected_homo[0]/projected_homo[2]
    y=projected_homo[1]/projected_homo[2]
    projected_points_2d = np.vstack((x,y)).T
    

    # raise NotImplementedError
    ##############################

    return projected_points_2d


def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    P = None

    start_time = time.time()

    ##############################
    # TODO: Student code goes here
    p0=initial_guess.flatten()[:-1]
    kwargs_dict={"pts2d":pts2d,"pts3d":pts3d}
    ls= least_squares(objective_func, p0, method='lm', verbose=2, max_nfev=50000, kwargs=kwargs_dict)
    P=np.array([[ls.x[0],ls.x[1],ls.x[2],ls.x[3]], [ls.x[4],ls.x[5],ls.x[6],ls.x[7]], [ls.x[8],ls.x[9],ls.x[10],1]])
    # raise NotImplementedError
    ##############################

    print("Time since optimization start", time.time() - start_time)

    return P


def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    K = None
    R = None

    ##############################
    # TODO: Student code goes here
    M=P[:,:-1]
    K, R = rq(M)
    # raise NotImplementedError
    ##############################

    return K, R


def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    cc = None

    ##############################
    # TODO: Student code goes here
    inv= np.linalg.inv(np.dot(K,R_T))
    I_t= np.dot(inv,P)
    cc=-I_t[:,-1]
    # raise NotImplementedError
    ##############################

    return cc
