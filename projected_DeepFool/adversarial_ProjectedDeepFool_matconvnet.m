%%
% This is a modification of standard DeepFool which confines the attack to
%   the subspace specified by the supplied (semi-orthogonal) basis matrix S.
% The mechanism is simply the projection of Df (gradient of targeted 
%   DeepFool objective w.r.t. input image) to the column space of S.
% This is for doing illustrative experiments on the relationship between
%   classification performance and adversarial vulnerability (i.e. to show
%   that attacks happen along the directions that the net uses to achieve
%   test accuracy).
% 
% Deviations from the original source (https://github.com/LTS4/DeepFool),
%   are noted with the initials SJ/NL.
% The only material change is the single line implementing the subspace 
%   confinement, as commented below.
% Other minor changes involve returning the clean and adversarial "confidences".
% 
% The original header for adversarial_DeepFool_matconvnet (the main file 
%   for standard DeepFool) is preserved below.
% This depends on adversarial_perturbation.m from the original DeepFool
%   source, also lightly modified as commented, and included with this
%   project.

%%
%   MATLAB code for DeepFool
%
%   adversarial_DeepFool_matconvnet(x,net):
%   computes the adversarial perturbations for a MatConvNet's model
%
%   INPUTS
%   x: image in W*H*C format
%   net: MatConvNet's network (without loss layer)
%   opts: A struct containing parameters (see README)
%   OUTPUTS
%   r_hat: minimum perturbation
%   l_hat: adversarial label
%   l: classified label
%   itr: number of iterations
%
%   please cite: S. Moosavi-Dezfooli, A. Fawzi, P. Frossard: DeepFool: a simple and accurate method to fool deep neural networks.
%   In Computer Vision and Pattern Recognition (CVPR 2016), IEEE, 2016.
%%

% SJ/NL: Function header changed to output adv_conf and clear_conf, and to
% take parameter S.
function [r_hat,l_hat,adv_conf,l,clean_conf,itr] = ...
    adversarial_ProjectedDeepFool_matconvnet(x, net, S, opts)
    
size_x = size(x);
x = reshape(x,numel(x),1);
out = f(x,0);
c = numel(out);
[clean_conf,l] = max(out); % SJ/NL: retaining and returning clean_conf

if nargin == 4 % SJ/NL: accounting for expanded parameter list
    adv = adversarial_perturbation(x,l,@Df,@f,opts);
else
    adv = adversarial_perturbation(x,l,@Df,@f);
end

l_hat = adv.new_label;
adv_conf = adv.new_conf; % SJ/NL: returning adv_conf
r_hat = reshape(adv.r,size_x);
itr = adv.itr;

    function out = f(y,flag)
        %do forward pass
        res = vl_simplenn(net,single(reshape(y,size_x)),[],[],'Mode','test');
        out = res(end).x(:)';
        
        %flag==0:compute the outputs        
        %flag==1:compute the label
        if flag==1
            [~,out] = max(out);
        end
    end

    function dzdx = Df(y,label,idx)        
        for i=1:numel(idx)
            
            dzdy = zeros(1,1,c,'single');
            dzdy(idx(i)) = 1;
                        
            %do forward-backward pass
            res = vl_simplenn(net,single(reshape(y,size_x)),dzdy,[],'Mode','test');
            dzdx(:,i) = reshape(res(1).dzdx,prod(size_x),1);
        end
        dzdx = dzdx-repmat(dzdx(:,idx==label),1,numel(idx));
        
        % SJ/NL: Projection of dzdx onto the subspace represented by S.
        dzdx = S * (S' * dzdx);
        % Now, dzdx contain only its components in the basis S: this is
        % equivalent to confining DeepFool's linear attack to S.        
    end

end


