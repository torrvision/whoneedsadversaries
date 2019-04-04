%%
% This function accepts a matrix of (e.g. adversarial attack) vectors, a 
%   collection of orthonormal (basis) directions and some information about
%   them, and a vector of subspace dimensions (i.e. the number of basis 
%   vectors to be retained in each trial). It outputs the norms of the 
%   projections of the input vectors onto each subspace as a fraction of 
%   the original norm (i.e. the norm of the projection of the corresponding
%   unit-norm direction vector).
%
% Inputs:
%   - perturbation_matrix: matrix of sample adversarial perturbations, as 
%       column vectors, expressed in the same coordinate frame as the basis 
%       vectors; this will typically be canonical image space, vectorised 
%       according to some convention (e.g. as by the Matlab reshape command)
%   - dir_info: a struct with two fields:
%       - dir_info.V: a semi-orthogonal basis matrix (i.e. with orthonomal 
%			columns; typically singular vectors)
%       - dir_info.D: a diagonal matrix of significance values (typically
%           singular values corresponding to the singular vectors V)
%   - vectors_per_trial: vector containing how many basis vectors should 
%       be retained to define the subspace for each trial (with 
%       length(vectors_per_trial) trials done per sorting strategy)
%   - selection_strategy: a string specifying the ordering used to select the
%       vectors for the output subspace basis, for each entry in 
%       vectors_per_trial; see the possible choices in
%       utilities/construct_subspace_basis.
% Outputs:
%   - projected_norm_mat: a matrix in which the row index corresponds to 
%       the subspace dimensions specified in vectors_per_trial, and the 
%       column index to the vectors input in perturbation_matrix; each
%       entry denotes the norm of the corresponding vector projected onto
%       the corresponding subspace, as a fraction of its original norm
% Note: "norm" means "2-norm" here; if desired, this can be changed by
%   changing the arguments to normalize and vecnorm.

%%
function projected_norm_mat = subspace_component_norms(perturbation_matrix, ...
    dir_info, vectors_per_trial, selection_strategy)   

    num_perturbations = size(perturbation_matrix, 2);
    num_trials = length(vectors_per_trial);
    projected_norm_mat = zeros(num_trials, num_perturbations);

    normalised_perturbations = normalize(perturbation_matrix, 'norm', 2);
    
    for i = 1:num_trials
       basis = construct_subspace_basis(dir_info, vectors_per_trial(i), selection_strategy);
       projected_normalised_perturbations = basis' * normalised_perturbations;
       projection_norms = vecnorm(projected_normalised_perturbations, 2, 1);      
       projected_norm_mat(i,:) = projection_norms;
    end
end
