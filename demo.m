%
%  Visual Tracking Using Attention-Modulated Disintegration and Integration
%
%  Jongwon Choi, 2016
%  https://sites.google.com/site/jwchoivision/
%  contact: jwchoi.pil@gmail.com
%  
%
%  Demo program of SCT4.
%  You can use this program freely for research and please acknowledge the paper[1].
%  You should contact to us for any commercial usage.
%  When you need the program of SCT6, please contact to the authors.
%
%  *** Piotr Dollar's toolbox[2] and some codes from Henriques et al.[3] were utilized.
%
%  [1] J. Choi, H. J. Chang, J. Jeong, Y. Demiris, J. Y. Choi, "Visual Tracking 
%      Using Attention-Modulated Disintegration and Integration", CVPR, 2016
%  [2]  P. Dollar, ¡°Piotr¡¯s Computer Vision Matlab Toolbox (PMT)¡±, 
%      http://vision.ucsd.edu/?pdollar/toolbox/doc/index.html.
%  [3]  J. F. Henriques, R. Caseiro, P. Martins, and J. Batista, ¡°HighSpeed Tracking 
%      with Kernelized Correlation Filters¡±, IEEE Transactions on PAMI, 2015
%
%

addpath('KCF');
addpath('strong');
addpath(genpath('PiotrDollarToolbox'));

% Inputs
base_path = 'Deer'; %dataset path
show_visualization = 1; %visualization option (0: not visible, 1: visible)

% Load the image data
[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path);

% Tracking start
% Position : [left-top-x left-top-y right-bottom-x right-bottom-y]
% time : computational time in second (without time for image load & visualization)
[positions, time] = sct4(video_path, img_files, pos, target_sz, show_visualization);
