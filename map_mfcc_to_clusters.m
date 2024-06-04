
% mapMfccToClusters.m

function observed_sequence = map_mfcc_to_clusters(mfcc_vectors, cluster_centers)
    num_vectors = size(mfcc_vectors, 1);
    observed_sequence = zeros(num_vectors, 1);
    
    for i = 1:num_vectors
        [~, closest_cluster] = min(pdist2(mfcc_vectors(i, :), cluster_centers));
        observed_sequence(i) = closest_cluster;
    end
end
