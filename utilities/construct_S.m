function reduced_Q = construct_S(curvature_info, vectors_per_class, selection_strategy)
% This takes a structure (from boundary_curvatures_all_targets,
% see that output format if you want to construct one yourself for whatever
% reason) of principal curvature information about the decision boundaries
% between each one of a chosen set of classes and all others, assembles a
% given number of these directions into a matrix by a given criterion, and 
% returns the span of that matrix as the reduced Q from the QR 
% decomposition (which is semi-orthogonal). The ordering of the curvatures on
% input is most positive to most negative.

% This matrix Q is primarily intended for use by other scripts that examine
% the effects of projecting images into this space (which plays the role of
% "S" in the Dezfooli/Fawzi reports) before classification.

% Each class contributes an equal number of direction vectors, and the 
% per-class number is what the user specifies. (Other approaches are
% possible, e.g. taking the N highest-magnitude negative curvatures
% from whichever classes they occur on.)
    num_classes = length(curvature_info.data);
    directions = [];
    for i=1:num_classes        
        directions = [directions construct_subspace_basis(curvature_info.data{i}, ...
            vectors_per_class, selection_strategy)];
    end
    [reduced_Q, ~] = qr(directions,0);
