%%
% This is a modification of standard DeepFool to force targeting of a
%   supplied class ID.
% This is as exactly as is done in the (Caffe) version of the code
%   originally provided personally by Seyed-Mohsen 
%   Moosavi-Dezfooli and Alhussein Fawzi.
% 
% This port derives from the original DeepFool source (https://github.com/LTS4/DeepFool),
%   with changes noted with the initials SJ/NL.
% This depends on adversarial_perturbation_fixed_label.m, included in the 
%   package: see comments there. 
%
% The original header for adversarial_DeepFool_matconvnet (the main file 
%   for standard DeepFool) is preserved below.
%
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

% NL/SJ: Function header changed to take parameter lab_adv (the index of
%   the target class).
function [r_hat,l_hat,l,itr] = ...
    adversarial_DeepFool_matconvnet_fixed_label(x, net, lab_adv, opts)
    
size_x = size(x);
x = reshape(x,numel(x),1);
out = f(x,0);
c = numel(out);
[~,l] = max(out);

if nargin == 4 % SJ/NL: accounting for expanded parameter list
    adv = adversarial_perturbation_fixed_label(x, l, @Df, @f, lab_adv, opts);
else
    adv = adversarial_perturbation_fixed_label(x, l, @Df, @f, lab_adv);
end

l_hat = adv.new_label;
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
    end

end
