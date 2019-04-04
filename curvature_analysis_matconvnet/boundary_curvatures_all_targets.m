%%
% Given a dataset (data) and a matconvnet SimpleNN network with its 
%   softmax layer removed (net), this function estimates the "mean 
%   principal directions and curvatures" for all of the net's binary
%   (1-v-all) class decision boundaries, in the vicinity of the dataset 
%   samples. The principal directions for the boundary between each
%   left-out class i and all others are returned as columns in the
%   semi-orthogonal output matrix res.data{i}.V, with the corresponding 
%   curvatures in the diagonal matrix res.data{i}.D.
%
% This sits on top of boundary_curvatures_one_target, eliminating 
%	its fixed_fool_lab parameter by implementing the leave-one-out loop
%	itself, and gathering all of the results into the output structure res.
%
% Please see the comments in boundary_curvatures_one_target for 
%	explanations of the numeric parameters num_eigs and cdiff_adj_fac, 
%	and for citation and attribution information.
%
% If desired, one can restrict consideration to a limited number of class
%   IDs, passed in as the vector of indeces opts.class_subset.
% Similarly, one can specify the number of sample training images to use in the
%	computation via opts.num_sample_images. (E.g. we used 100 in our own experiments.)
% Anything else passed in through opts will get written into the result structure.
%   This can be handy for your record-keeping: e.g. we recommend writing 
%   info about exactly which dataset and net you were using; you can call 
%   your fields whatever you want, other than "class_subset" and "num_sample_images".
%

%%
function res = boundary_curvatures_all_targets(data, net, num_eigs, cdiff_adj_fac, opts)

	class_subset = 1:data.labels_limit; % Class subset defaults to all class IDs.
	num_sample_images = length(data.train_labels); % Number of sample images defaults to whole training set.

    if nargin == 5
        res.meta.opts = opts;
        if isfield(opts, 'class_subset')
            class_subset = opts.class_subset;
        end
		if isfield(opts, 'num_sample_images')
			num_sample_images = opts.num_sample_images;
		end
    else
        if nargin ~= 4
            error('Wrong number of arguments.');
        end
    end

    res.data = cell(length(class_subset),1);
    res.meta.num_eigs = num_eigs;
    res.meta.cdiff_adj = cdiff_adj_fac;

    for i=1:length(class_subset)
        [res.data{i}.V, res.data{i}.D, res.data{i}.flag] = ...
            boundary_curvatures_one_target(data, net, class_subset(i), ...
				num_eigs, cdiff_adj_fac, num_sample_images);
    end
