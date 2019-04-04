function [S_comp] = project_to_basis_subset(basis, n, input)
% This is a friendly wrapper around a projection operation to make what's
% being done a bit clearer to the user and reader, in the context of this
% project.

% Given a list of vectors as columns of a matrix (representing directions 
% in image space, in our case), and a semi-orthogonal basis, we just return
% those vectors projected onto that basis, AKA expressed in terms of it. 
% One twist: we allow the user to supply an index that truncates the basis 
% at a particular vector, thus performing dimensionality reduction. The 
% idea is that the user will have sorted the input basis in an order 
% meaningful to them: the truncation of the basis when sorted in order of
% signed curvature is what is referred to as "S" in the Dezfooli/Fawzi tech
% reports. Note that the code assumes, but does not assert, that the basis
% is actually semi-orthogonal (i.e. that all columns are mutually 
% orthonormal): it must be.

% Because the basis is semi-orthogonal, it's entirely appropriate to take the
% norms of the expressions in the new basis directly, in terms of the 
% output components of this function. You can e.g. compare the norm of the 
% input vector to the norm of the output vector to see the fraction of it 
% that landed in the (truncated) basis.
    S_comp = basis(:,1:n)' * input;
