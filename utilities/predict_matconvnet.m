function out = predict_matconvnet(ims, net, batchsize, gpu, num_cl, ...
    conf_flag, getDataset, add_image)
% This is a heavily modified version of the predict_matconvnet utility
% function available at
% https://github.com/LTS4/universal/blob/master/matlab/predict_matconvnet.m.
% It's used to evaluate a given net on a given set of images, possibly
% under a supplied (universal) additive perturbation.
% The user can optionally retain the full vector of "confidences" (outputs
% of the final softmax layer) and/or the vector of "logits" (the inputs to
% the final softmax layer). (Note that those terms have been placed between
% quote marks: no responsibility assumed here.)
% ims - collection of images; may be specified by paths in a cell
%   array/matrix, or as a 4D tensor: see parsing code below
% net - matconvnet net; must be of simpleNN type in this implementation
% batchsize - the batch size for evaluation
% gpu - logical to specify whether to move net and images to GPU for
%   evaluation: function assumes these come in on CPU, and returns them on
%   exit in any case
% num_cl - number of classes: length of the net's output vector
% conf_flag - option code specifying the information to be returned in
%   output structure: see code
% getDataset - (required) function for accessing images from a collection:
%   defined by the loader used to load the set originally
% add_image - optional additive universal perturbation to apply to all
%   images fed to the net

    if nargin == 8
        constant_add = add_image;
    else
        constant_add = 0;
    end        
    
    % Images may be supplied as 4D tensors (w/ images indexed along the 4th
    % dim), or as cell arrays of their paths.
    % Note that tensors of MNIST images must be supplied with a singleton
    % 3rd dimension (representing the single channel).
    if iscell(ims)
        num_images = size(ims,ndims(ims));
    else
        if ndims(ims) == 4
            num_images = size(ims,4);
        elseif ndims(ims) < 4
            num_images = 1;            
        end
    end
    
    ys = zeros(num_images, 1);
    conf = zeros(num_images, num_cl);
    scores = zeros(num_images, num_cl);
        
    if gpu
        net = vl_simplenn_move(net, 'gpu');
    end
    
    for b = 1:batchsize:num_images
        b_end = min( (b+batchsize-1), num_images );
        batch_ims = getDataset(ims, b:b_end);

        if constant_add ~= 0
            if ndims(batch_ims) < 4 % then batch_ims is actually a single image.
                batch_ims = single( batch_ims + constant_add );
            else
                batch_ims = single( batch_ims + repmat(constant_add, [1 1 1 size(batch_ims,4)]) );
            end
        else
            batch_ims = single(batch_ims);
        end            
        
        if gpu
            batch_ims = gpuArray(batch_ims);
        end
        
        res = vl_simplenn(net, batch_ims, [], [], 'Mode', 'test');
        [max_conf(b:b_end), ys(b:b_end)] = max(squeeze(gather(res(end).x)));
        if conf_flag == 2
            conf(b:b_end,:) = squeeze(gather(res(end).x))';
            scores(b:b_end,:) = squeeze(gather(res(end-1).x))';
        end
        
        % Assisting MATLAB's garbage collector.
        batch_ims = [];
    end
    
    % Assisting MATLAB's garbage collector.
    net = [];
    
    switch conf_flag
        case 0
            out = ys;    
        case 1
            out{1} = ys;
            out{2} = max_conf;    
        case 2
            out{1} = ys;
            out{2} = max_conf;
            out{3} = scores;
            out{4} = conf;    
        otherwise
            error('Unrecognised value of conf_flag.');
    end                
    
end

