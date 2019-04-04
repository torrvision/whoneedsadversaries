%%
% See comments in boundary_curvatures_one_target.
% 
% This is the extension of gradient_net_matconvnet to handle multiple input
%   images at once. See the comments in that file.
% 
% The input images xs are assumed to come in the usual HxWxCxI format, with
%   classes_orig and classes_fooled as corresponding column I-vectors.

function grad_xs = gradient_net_multiple_matconvnet(xs, net, classes_orig, classes_fooled, gpu)

    assert( size(xs,4) == length(classes_orig) );
    assert( length(classes_orig) == length(classes_fooled) );
    assert( size(classes_orig, 2) == 1 );
    assert( size(classes_fooled, 2) == 1 );

    if gpu>0
        net = vl_simplenn_move(net, 'gpu');
        xs = gpuArray(xs) ;
    end

    num_images = size(xs,4);
    num_classes = length(net.meta.classes.name);
    
    dzdy = zeros(1, 1, num_classes, num_images);
    ones_vector = ones(num_images, 1);
    dzdy( sub2ind(size(dzdy), ones_vector, ones_vector, classes_orig, ...
        (1:num_images)') ) = 1; % A unit of increase in original class score is a unit of increase in energy.
    dzdy( sub2ind(size(dzdy), ones_vector, ones_vector, classes_fooled, ...
        (1:num_images)') ) = -1; % a unit of increase in fooling class score is a unit of decrease in energy.
    dzdy = single(dzdy);
    if gpu>0
       dzdy = gpuArray(dzdy); 
    end
    
    res = vl_simplenn(net, xs, dzdy, [], 'Mode', 'test');
    
    grad_xs = res(1).dzdx;
        
    if gpu>0
        grad_xs = gather(grad_xs);
        % Try to help the garbage collector to do its job:
        res = [];
        net = [];
        xs = [];
        dzdy = [];
    end
    
end
