%%
% This function collects information about "confined/projected" DeepFool
%   perturbations over a dataset (or a random subset of it), for a given
%   net. It takes a semi-orthogonal matrix of singular vectors and the
%   corresponding diagonal matrix of singular values, and performs this 
%   evaluation for subspaces spanned by the most and least significant
%   singular vectors: the subspace dimensions (numbers of singular vectors
%   spanning the subspace in each evaluation) of all evaluations to be
%   performed are specified by the user in an argument vector. Information
%   is saved to a file rather than returned (due to potential size issues).
% Details of these parameters are as follows:
%   - data: dataset as returned by a loader function in the utility folder
%   - net: MatConvNet simpleNN, with the softmax layer removed
%   - dir_info: a struct with two fields:
%       - dir_info.V: a basis matrix of orthonormal columns (typically
%           singular vectors)
%       - dir_info.D: a diagonal matrix of significance values (typically
%           singular values corresponding to the singular vectors V)
%   - vectors_per_trial: vector containing how many basis vectors from
%       dir_info.V should be retained to define the subspace for each trial
%       (with length(vectors_per_class) trials done per sorting strategy)
%   - output_filename: the stem of the file to which the result structures
%       will be saved, including path information if desired; a good
%       idea is to save identifying information of the net and dataset
%       being evaluated: the strategy and subspace dimension will be
%       appended for each output structure
%   - sample_counts: an optional 2-vector [train_samples test_samples]
%       specifying how many random samples to draw from each of the train
%       and test sets (if one wants to do less computation)
%
% See adversarial_ProjectedDeepFool_matconvnet for further explanation of
%   what is meant by "confinement/projection" in this context, including
%   the modification that implements it.

%%
function deepfool_projected_vs_subspace(data, net, dir_info, ...
    vectors_per_trial, output_filename, sample_counts)

    sample_data = data;
    if nargin == 6 % If the user has supplied the sample_counts vector...
        random_train_indeces = randperm(length(data.train_labels), sample_counts(1));
        random_test_indeces = randperm(length(data.test_labels), sample_counts(2));        
        sample_data.train_images = data.getDataset(data.train_images, random_train_indeces);
        sample_data.test_images = data.getDataset(data.test_images, random_test_indeces);
        sample_data.train_labels = data.train_labels(random_train_indeces);
        sample_data.test_labels = data.test_labels(random_test_indeces);
    elseif nargin ~= 5
        error('Invalid argument count.');
    end

    selection_strategies = {'least', 'most'};

    for main_ind = 1:length(selection_strategies)
        
        strategy = selection_strategies{main_ind};
        num_trials = length(vectors_per_trial);
        
        for i = 1:num_trials            
            basis = construct_subspace_basis(dir_info, vectors_per_trial(i), ...
                strategy);
           
            [train_set_results, test_set_results] = deepfool_projected(...
                sample_data, net, basis);                        
            
            save(sprintf('%s_%s_%d_df_confined.mat', output_filename, ...
               strategy, vectors_per_trial(i)), 'train_set_results',...
               'test_set_results', '-v7.3');
        end
    end
    
end
