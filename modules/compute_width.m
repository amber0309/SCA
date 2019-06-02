function md = compute_width(dist_mat)
    % input is the squared distance matrix, i.e. [dist_mat]_{ij} = || x_{i} - x{j} ||^2
    % output is for exponential kernel with 2*sgm*sgm in denominator

    half_dist = dist_mat-tril(dist_mat);
    half_dist = reshape(half_dist, size(dist_mat, 1)*size(dist_mat, 2), 1);
    md = sqrt(0.5*median(half_dist(half_dist>0)));

end