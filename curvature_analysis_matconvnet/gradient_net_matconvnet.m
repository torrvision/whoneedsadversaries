%%
% See comments in boundary_curvatures_one_target.
% 
% Given source (class_orig) and target (class_fooled) class indeces, and a
%   net with the softmax layer removed (net), computes the gradient of the
%   difference between the source and target class scores at the input image
%   (x).
% If a positive number of gpus is indicated (gpu), the computation takes
%   place on the GPU. Inputs are assumed to come in CPU form, and are
%   returned accordingly.

%%
function grad_x = gradient_net_matconvnet(x, net, class_orig, class_fooled, gpu)
    
    if gpu > 0
        net = vl_simplenn_move(net, 'gpu');
        x = gpuArray(x) ;
    end
    
    % The function that we want to differentiate is the difference between
    %   the score of the original class and the score of the (specified)
    %   fooling class. So our dzdy (derivative of the objective w.r.t. the 
    %   net's output vector) is very simple:
    num_classes = length(net.meta.classes.name);
    dzdy = zeros(1, 1, num_classes);
    dzdy(class_orig) = 1; % A unit of increase in original class score is a unit of increase in energy.
    dzdy(class_fooled) = -1; % A unit of increase in fooling class score is a unit of decrease in energy.
    dzdy = single(dzdy);
    if gpu>0
       dzdy = gpuArray(dzdy); 
    end
        
    res = vl_simplenn(net, x, dzdy, [], 'Mode', 'test');
    
    grad_x = res(1).dzdx;
        
    if gpu>0
        grad_x = gather(grad_x);
        % Try to help the garbage collector to do its job:
        res = [];
        net = [];
        x = [];
        dzdy = [];
    end
    
end
