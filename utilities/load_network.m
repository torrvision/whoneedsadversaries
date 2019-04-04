function [net, net_name] = load_network(net_path, net_type)
% Loads and tidies (for matconvnet) a net file onto the CPU.
% Supply the name of the network file, including whatever path information
% is required, as the string net_path.
% If specified, the string net_type option will perform additional 
% processing of the net before deployment, for matconvnet's sake. See the 
% switch statement.
    [~, net_name, ~] = fileparts(net_path);
    load(net_path, 'net');
    net = vl_simplenn_tidy(net); % add compatibility to newer versions of MatConvNet
    net = vl_simplenn_move(net, 'cpu');        
    if nargin > 1
        switch net_type
            case 'CIFAR'
                net = simpleMergeBatchNorm(net);
                net = simpleRemoveLayersOfType(net, 'bnorm');
            case 'ImageNet'
                net = cnn_imagenet_deploy(net);
            otherwise
                warning('Unknown net type passed to load_network.');
        end
    end
end


% -------------------------------------------------------------------------

% The below subroutines have been copied over from matconvnet's
% cnn_imagenet_deploy function:

% -------------------------------------------------------------------------
function net = simpleMergeBatchNorm(net)
% -------------------------------------------------------------------------
    for l = 1:numel(net.layers)
        if strcmp(net.layers{l}.type, 'bnorm')
            if ~strcmp(net.layers{l-1}.type, 'conv')
              error('Batch normalization cannot be merged as it is not preceded by a conv layer.') ;
            end
            [filters, biases] = mergeBatchNorm(...
              net.layers{l-1}.weights{1}, ...
              net.layers{l-1}.weights{2}, ...
              net.layers{l}.weights{1}, ...
              net.layers{l}.weights{2}, ...
              net.layers{l}.weights{3}) ;
            net.layers{l-1}.weights = {filters, biases} ;
        end
    end
end

% -------------------------------------------------------------------------
function [filters, biases] = mergeBatchNorm(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
    a = multipliers(:) ./ moments(:,2) ;
    b = offsets(:) - moments(:,1) .* a ;
    biases(:) = biases(:) + b(:) ;
    sz = size(filters) ;
    numFilters = sz(4) ;
    filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;
end

% -------------------------------------------------------------------------
function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
    layers = simpleFindLayersOfType(net, type) ;
    net.layers(layers) = [] ;
end

% -------------------------------------------------------------------------
function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
    layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;
end
