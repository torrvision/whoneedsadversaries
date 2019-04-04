%%
% See comments in boundary_curvatures_one_target.
% 
% Given source (class_orig) and target (class_fooled) class index column
%   vectors, and a net with the softmax layer removed (net), numerically
%   estimates Hv at supplied image point(s) x and direction vector v, where 
%   H is the sample mean Hessian (over the supplied points x) of the 
%   difference between the net's source and target class score functions.
% Uses a basic (central) finite difference method, with perturbation
%   coefficient h.
% See gradient_net_multiple_matconvnet for further clarification of input
%   format.
%
% This file is a loose port of code originally supplied personally by
%   Seyed-Mohsen Moosavi-Dezfooli and Alhussein Fawzi.

function Hv = hessian_numerical_matconvnet(x, v, net, h, class_orig, ...
    class_fooled, gpu)

    % NOTE: Because of the fact that we're interested in exerting control
    % over the sizes of perturbations used in estimating boundary
    % curvature, based on prior assumptions about the sorts of scales
    % relevant to UAP behaviour, we make sure to control not only the h 
    % input here but also the scale of the input vector over the course of
    % the internal numerical approximations.
    % If a non-unit scale was applied to v on input, that will be
    % multiplied back into Hv before returning it, but it won't be allowed
    % to interfere with the scale being used in the meantime.
    v_scale = norm(v);
    v = v / v_scale;

    sample_size = size(squeeze(x(:,:,:,1)));
    num_samples = size(x,4);
    
    x_fwd_prt = bsxfun(@plus, x, h*reshape(v, sample_size));
    dzdx_fwd_prt = gradient_net_multiple_matconvnet(x_fwd_prt, ...
        net, class_orig, class_fooled, gpu);
    dzdx_fwd_prt_sample_mean = mean( reshape(dzdx_fwd_prt, prod(sample_size), ...
        num_samples), 2 );
    
    x_bwd_prt = bsxfun(@plus, x, -h*reshape(v, sample_size));
    dzdx_bwd_prt = gradient_net_multiple_matconvnet(x_bwd_prt, ...
        net, class_orig, class_fooled, gpu);
    dzdx_bwd_prt_sample_mean = mean( reshape(dzdx_bwd_prt, prod(sample_size), ...
        num_samples), 2 );
    
    Hv = ( dzdx_fwd_prt_sample_mean - dzdx_bwd_prt_sample_mean ) / (2*h);
    
    % NOTE: See corresponding comment above.
    Hv = Hv * v_scale;
    
    Hv = double(Hv); % Was getting grief from the eigs function in 
                     %  Matlab 2018b when this was returned as a single.
end
