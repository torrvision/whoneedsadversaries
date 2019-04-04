function subspace_basis = construct_subspace_basis(dir_info, num_vectors, selection_strategy)
% Outputs a particular semi-orthogonal subspace basis, given a collection of
% orthonormal directions, some information about them, and some
% specifications dictating selection of the subspace basis vectors.

% dir_info: dir_info.V is a semi-orthogonal matrix, and dir_info.D is a
%   diagonal matrix of "scores" of the corresponding column vectors in
%   dir_info.V. In the context of this project, V and D can represent
%   principal directions and curvatures (from the second-order analysis), or
%   singular vectors and values (from the first-order analysis). It's
%   assumed that they're ordered from positive to negative.
% num_vectors: The number of basis vectors to select, according to
%   selection_strategy (i.e. the dimension of the output subspace, though
%   see the comment on the first_and_last option.
% selection_strategy: a string specifying the ordering used to select the
%   top num_vectors vectors for the output subspace basis. See definitions
%   in the switch statement.
    switch selection_strategy
        case 'first' % first N (most positive sing. val. / curv.) directions
                % (same as 'most' for singular values)
            subspace_basis = dir_info.V(:,1:num_vectors);
        case 'last' % last N (most negative sing. val. / curv.) directions
                % (same as 'least' for singular values)
            thisV = fliplr(dir_info.V);
            subspace_basis = thisV(:,1:num_vectors);
        case 'most' % N highest-magnitude (sing. val. / curv.) directions
            [~, inds] = sort(abs(diag(dir_info.D)), 'descend');       
            subspace_basis = dir_info.V(:,inds(1:num_vectors));
        case 'least' % N lowest-magnitude (sing. val. / curv.) directions
            [~, inds] = sort(abs(diag(dir_info.D)), 'ascend');       
            subspace_basis = dir_info.V(:,inds(1:num_vectors));
        case 'first_and_last' % N most positive *and* N most negative directions
                % Note that you get 2N directions total in this case.
            subspace_basis = dir_info.V(:,1:num_vectors);
            thisV = fliplr(dir_info.V);
            subspace_basis = [subspace_basis thisV(:,1:num_vectors)];    
        otherwise
            error('You chose a selection strategy that hasn''t been implemented.');
    end
