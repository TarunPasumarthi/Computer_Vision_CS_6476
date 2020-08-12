import numpy as np
import math
from proj2_code.least_squares_fundamental_matrix import solve_F
from proj2_code import two_view_data
from proj2_code import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    # TODO: Student code goes here
    num_samples= math.log(1-prob_success)/math.log(1-ind_prob_correct**sample_size)
    # raise NotImplementedError
    ##############################

    return num_samples


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    inliers = None

    ##############################
    # TODO: Student code goes here
    il= []
    if(x_0s.shape[1]==2):
        x_0s=np.hstack((x_0s,np.ones((x_0s.shape[0],1))))
        x_1s=np.hstack((x_1s,np.ones((x_1s.shape[0],1))))
    for i in range(len(x_0s)):
        d1= np.abs(fundamental_matrix.point_line_distance(np.dot(F,x_1s[i]),x_0s[i]))
        if(d1<=threshold):
            il.append(i)
    inliers=np.array(il)
    # raise NotImplementedError
    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliears_x_0 = None
    inliears_x_1 = None

    ##############################
    # TODO: Student code goes here
    iterations= int(np.ceil(calculate_num_ransac_iterations(.99,9,.9)))
    for i in range(iterations):
        # randomly select subset
        idx= np.random.choice(len(x_0s),9)
        tempx0s= x_0s[idx]
        tempx1s= x_1s[idx]
        
        #fit a model to the subset
        tempF= solve_F(tempx0s,tempx1s)
        
        #determine inliers and replace F if better model
        i_idx= find_inliers(x_0s, tempF, x_1s, 1)
        if(i==0 or len(inliears_x_0)<len(i_idx)):
            best_F=tempF
            inliears_x_0= x_0s[i_idx]
            inliears_x_1= x_1s[i_idx]
        
    # raise NotImplementedError
    ##############################

    return best_F, inliears_x_0, inliears_x_1
