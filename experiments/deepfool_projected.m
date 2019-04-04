%%
% Given a dataset (data), a matconvnet SimpleNN network with its 
%   softmax layer removed (net), and a semi-orthogonal basis matrix (basis),
%   this function computes the "confined DeepFool" at each point in the
%   dataset, where basis defines the subspace of confinement.
% This is mostly a convenient wrapper around adversarial_ProjectedDeepFool_matconvnet,
%   which processes an entire dataset and outputs detailed result
%   structures train_set_results and test_set_results (containing the
%   fields below) for subsequent experimental analysis.

%%
function [train_set_results, test_set_results] = deepfool_projected(data, net, basis)

    % DeepFool settings:
    opts.labels_limit = data.labels_limit;
    opts.overshoot = 0.02;

    train_image_count = size(data.train_labels, 2);
    test_image_count = size(data.test_labels, 2);
    train_set_results = cell(train_image_count, 1);
    test_set_results = cell(test_image_count, 1);
       
    for i = 1:train_image_count
        %if (rem(i,1000)==0), tic; end
        x = feval(data.getDataset, data.train_images, i);
        [r, adversarial_label, adv_conf, clean_label, clean_conf, itr] = ...
            adversarial_ProjectedDeepFool_matconvnet(x, net, basis, opts);
        train_set_results{i} = struct('index', i, 'image', x, ...
            'adversarial_pert', r, 'adversarial_label', adversarial_label, ...
            'adv_conf', adv_conf, 'clean_label', clean_label, ...
            'clean_conf', clean_conf, 'iterations', itr, ...
            'ground_label', data.train_labels(i)); 
        %if (rem(i,1000)==0),toc; end
    end

    for i = 1:test_image_count        
        x = feval(data.getDataset, data.test_images, i);
        [r, adversarial_label, adv_conf, clean_label, clean_conf, itr] = ...
            adversarial_ProjectedDeepFool_matconvnet(x, net, basis, opts);
        test_set_results{i} = struct('index', i, 'image', x, ...
            'adversarial_pert', r, 'adversarial_label', adversarial_label, ...
            'adv_conf', adv_conf, 'clean_label', clean_label, ...
            'clean_conf', clean_conf, 'iterations', itr, ...
            'ground_label', data.test_labels(i));
    end

    % Convert from cell arrays to struct arrays:
    train_set_results = [train_set_results{:}];
    test_set_results = [test_set_results{:}];

end
