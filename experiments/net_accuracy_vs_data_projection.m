%%
% Given a dataset (data), a matconvnet SimpleNN network with its 
%   softmax layer removed (net), a semi-orthogonal basis matrix (basis), and a
%   number (n) of basis vectors to retain in the subspace (from the left of
%   the matrix towards the right), this function computes the train- and 
%   test-set accuracies of the net when evaluated on the data projected 
%   onto the specified subspace. The optional logical argument gpu specifies
%	whether to perform the evaulation on the GPU (default: false).
% The output structure res contains two fields:
%   - accuracies: a struct of the type output by test_perf: see comments
%       there
%   - data_projected: the input dataset (data), but with the images replaced
%       by their projected counterparts (illustrating what the net was
%       actually evaluated on)

%%
function res = net_accuracy_vs_data_projection(data, net, basis, n, gpu)

if nargin == 4
	gpu = false;
elseif nargin ~= 5
	error('Wrong number of arguments.');
end

add_image = 0;
score_function = @(data_arg)test_perf(data_arg, net, gpu, data.getDataset, add_image);

image_size = data.image_size;
num_train_images = length(data.train_labels);
num_test_images = length(data.test_labels);

train_image_mat = reshape(data.train_images, [prod(image_size) num_train_images]);
test_image_mat = reshape(data.test_images, [prod(image_size) num_test_images]);

training_set_S = project_to_basis_subset(basis, n, train_image_mat);
test_set_S = project_to_basis_subset(basis, n, test_image_mat);

% As we're implementing dimensionality reduction via projection, one might
%   argue that the best thing one can do is to take the mean component
%   in S_perp over the dataset and add that back to every image before
%   classification.
% Alternatively, we can leave all of the components in S_perp projected
%   away to zero and see what happens.
reconstitute_perp_mean = true;

% Express the projections back in the canonical (image-space) basis.
projected_training_set_canon = basis(:,1:n) * training_set_S;
projected_test_set_canon = basis(:,1:n) * test_set_S;

% Add the mean component in S_perp back to all of the samples, if you've
%   chosen to do that.
if reconstitute_perp_mean
   perp_training_set_canon = train_image_mat - projected_training_set_canon;
   training_set_S_perp_mean_canon = mean(perp_training_set_canon, 2);   
   projected_training_set_canon = projected_training_set_canon + ...
       (training_set_S_perp_mean_canon * ones(1, size(projected_training_set_canon,2)));
   % Note that we are using the S_perp mean for the *training* set for
   %    adjusting the test set (as one would have to).
   projected_test_set_canon = projected_test_set_canon + ...
       (training_set_S_perp_mean_canon * ones(1, size(projected_test_set_canon,2)));
end

data_projected = data;
data_projected.train_images = single(reshape(projected_training_set_canon, size(data.train_images)));
data_projected.test_images = single(reshape(projected_test_set_canon, size(data.test_images)));

res.accuracies = score_function(data_projected);
res.data_projected = data_projected;
