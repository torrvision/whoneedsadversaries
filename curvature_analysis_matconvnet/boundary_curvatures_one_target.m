
%%
% Given a dataset (data), a matconvnet SimpleNN network (net) with its 
%   softmax layer removed (net), and a single class ID (fixed_fool_lab),
%   this function estimates the "mean principal directions and curvatures"
%   of the net's binary (1-v-all) decision boundary between the supplied 
%   class ID and all other classes, in the vicinity of the dataset samples.
%   The principal directions are returned as columns in the semi-orthogonal
%   output matrix V, with the corresponding curvatures in the diagonal
%   matrix D.
%
% The user also supplies two parameters which govern the numerics of the
%   computation:
% num_eigs - The number of (largest magnitude) Hessian eigenvalues to 
%   numerically estimate, i.e. the number of principal directions/curvatures 
%   to compute. NOTE: This may need to be less than the rank of the Hessian 
%   to force the use of the appropriate numerical method by MATLAB, depending 
%   on the version. E.g. for MNIST, the rank would be 784, and so 783 would 
%   be the maximum number of directions to compute (the final direction, 
%   of course, recoverable as the orthogonal complement if required).
% cdiff_adj_fac - A multiplier controlling how large the central-difference
%   steps used in Hessian estimation are. See the computation of 
%   central_difference_step_size in the code for details. (The value used
%   in the experiments in the paper was 10.)
% Additionally, the number of images to be subsampled (from the training set)
%	can be passed as the optional parameter num_sample_images, to save 
%	computation time.
%
% For further details and explanation of the significance of this analysis,
%   see the paper "With Friends Like These, Who Needs Adveraries?"; by
%   Saumya Jetley*, Nicholas A. Lord*, and Philip H.S. Torr; in NeurIPS
%   2018. Please cite that work if using this code.
%
% The subproject associated with this file, curvature_analysis_matconvnet,
%   is a port of research code personally provided by Seyed-Mohsen 
%   Moosavi-Dezfooli and Alhussein Fawzi. Matconvnet has been substituted
%   for Caffe, utility functions have been reimplemented, and other minor 
%   changes have been made, but essential functionality is preserved. The
%   relevant work (which our paper extends), which should also be cited,
%   is:
% Robustness of Classifiers to Universal Perturbations: A Geometric
%   Perspective; by Seyed-Mohsen Moosavi-Dezfooli*, Alhussein Fawzi*, Omar 
%   Fawzi, Pascal Frossard, and Stefano Soatto; in ICLR 2018.

%%
function [V, D, flag] = boundary_curvatures_one_target(data, net, fixed_fool_lab, ...
    num_eigs, cdiff_adj_fac, num_sample_images)

% Set the DeepFool options.
opts_deepfool.norm_p = 2;
opts_deepfool.max_iter = 5;

num_classes = data.labels_limit;
gpu_batch = true; % whether to process image batches on the GPU
gpu_single = false; % whether to process single images on the GPU
batchsize_single = 1; % just to denote that a single-image batch has size 1

% We consider only the training images here.
image_size = data.image_size;
images = single(data.train_images);
train_image_count = length(data.train_labels);

% Randomly choose a subset of the images.
if nargin == 5
	num_sample_images = train_image_count;
elseif nargin ~= 6
	error('Wrong number of arguments.');
end
sample_indeces = randperm(train_image_count, num_sample_images);

% Initialise an empty matrix whose columns will be perturbations to the
% fixed fooling class.
rs = [];

% Initialise empty containers for...
% (i) the original label;
l_ts = [];
% (ii) the fooling label (which should only ever be the fixed fooling
% label, in this setup);
l_fs = [];
% (iii) the point at which the decision function gradient was computed;
sample_points = [];
% (iv) the decision function gradient;
grad_xs = [];
% (v) and the normalised decision function gradient (direction).
ns = [];

% Initialise the count of how many successful DeepFools there have been.
% (This is bounded above by num_sample_images. For instance, we discard
% samples that are already predicted to have label fixed_fool_lab.)
sample_cnt = 0;

for i = 1:num_sample_images    
    % If the net's not already predicting that random image to be of the
    % fixed fooling label...
    sample_image = images(:,:,:,sample_indeces(i));
    if predict_matconvnet(sample_image, net, batchsize_single, gpu_single, ...
            num_classes, 0, data.getDataset) ~= fixed_fool_lab
        % .. then try to fool that image into that label.
        [r1,l_f,l_t] = adversarial_DeepFool_matconvnet_fixed_label(...
            sample_image, net, fixed_fool_lab, opts_deepfool);
        % If you've now fooled to the fixed fooling class...
        if l_f == fixed_fool_lab
            % Record the existence of this example.
            sample_cnt = sample_cnt + 1;
            % Collect the fooling perturbation as a new column in the
            % fooling pertubation collection matrix.
            rs = [rs, r1(:)];
            % Step back a little bit across the fooling boundary (by
            % shortening the DeepFool perturbation a bit)...
            sample_point = sample_image + 0.8*r1;
            % ... and compute the gradient of the decision function between
            % the originally predicted class and the fool class at that
            % point.
            grad_x = gradient_net_matconvnet(sample_point, net, l_t, l_f, ...
                gpu_single);
            grad_x = grad_x(:);
            
            % Record the original label, fooling label (which should only
            % ever be the fixed fooling label, in this setup...), the point
            % at which the decision function gradient was computed, the
            % gradient, and the normalised gradient (direction).
            l_ts(sample_cnt, 1) = l_t;
            l_fs(sample_cnt, 1) = l_f;
            sample_points(:,:,:,sample_cnt) = sample_point;
            ns(:,sample_cnt) = grad_x / norm(grad_x(:));
            grad_xs(:,sample_cnt) = grad_x;            
        end
    end
end

% The below line represents a heuristic attempt to make a reasonable small
%   perturbation that works irrespective of the image size and gamut.
% Note that the concept of "works" being implemented here relies on the
%   idea that you're probably interested in keeping the *per-pixel*
%   perturbations roughly constant. It's a very "Inf-norm-flavoured" version
%   of a 2-norm normalisation, if that makes any sense.
central_difference_step_size = sqrt(prod(image_size)) * ...
    ( max(images(:)) - min(images(:)) ) / 512 * cdiff_adj_fac; 

sample_points = single(sample_points);
hess_mat_vector_proj = @(v) ( hessian_numerical_matconvnet(sample_points, ...
    single(v), net, central_difference_step_size, l_ts, l_fs, gpu_batch) );

opts_eigs.issym = 1;
opts_eigs.isreal = 1;
opts_eigs.tol = 1e-5;
opts_eigs.issym = true;
tic;
disp(['Computing eigendecomposition for target class ', num2str(fixed_fool_lab)]);
[V, D, flag] = eigs(hess_mat_vector_proj, prod(image_size), num_eigs, ...
    'lm', opts_eigs);
disp(['eigs returned code ', num2str(flag)]);
toc;

% To handle the fact that different MATLAB versions may return different
%   sort orders, we explicitly (and possibly redundantly) sort the
%   eigenvector and eigenvalue matrices in descending order:
[sorted_diag, sorted_inds] = sort(diag(D), 'descend');
V = V(:,sorted_inds);
D = diag(sorted_diag);
