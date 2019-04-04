%%
% This function takes a tensor of DeepFool perturbations (in (h,w,c,i)
%   format), and returns the matrices of (right) singular vectors (V) and
%   singular values (diagonal D) of the matrix obtained by vectorising the
%   DeepFool images, normalising them, and stacking them (as rows).
% This method is an complement/alternative to the more complicated analysis
%   done in the curvature_analysis_matconvnet project.
% The returned matrices provide a measure of the extent to which DeepFool
%   (and first-order adversarial attacks more generally) "prefer" certain
%   directions in image space. As explained in the below reference, these
%   are in turn intimately connected to the directions relied on by
%   classifiers for making classification decisions generally.
%
% For further details and explanation of the significance of this analysis,
%   see the paper "With Friends Like These, Who Needs Adveraries?"; by
%   Saumya Jetley*, Nicholas A. Lord*, and Philip H.S. Torr; in NeurIPS
%   2018. Please cite that work if using this code.

%%
% The function takes the full path to a mat file containing the tensor of
%   DeepFools described above, and the name of the variable in that mat
%   file representing the tensor. It should be possible to pass these
%   either as character arrays or strings.
function [V, D] = svd_deepfool_pert(deepfool_tensor_mat_path, deepfool_tensor_name)   

    df_perts = load(deepfool_tensor_mat_path, deepfool_tensor_name);
    df_perts = df_perts.(deepfool_tensor_name);
    [h,w,c,~] = size(df_perts);
    df_perts = reshape(df_perts, h*w*c, []);
    [pert_dim, pert_count] = size(df_perts);
    
    % Performing length (2-norm) normalisation of the perturbations. One
    %   can alternatively omit this step. (We did not note a significant
    %   effect on the results of the experiments in this paper either way.)
    df_perts = normalize(df_perts, 'norm', 2);
    
    % Conceptually, the SVD can be thought of as PCA without mean
    %   subtraction. For our purposes, we are interested in variance around
    %   the origin (zero perturbation). 
    if pert_count > pert_dim
        [V,D,~] = svd(df_perts * df_perts'); % <- Faster, at least on some Matlab versions.
        D = sqrt(D);
    else
        [V,D,~] = svd(df_perts, 'econ');
    end
    
end
