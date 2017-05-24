# Radial Basis Function neural network

## Introduction
https://www.youtube.com/watch?v=OUtTI99uRf4

## Algorithm

1. Use clustering (K-MEANS) for finding centers (radial neurons - hidden layer)
2. Normalization (K-NEIGHBOURHOODS) to choose spreads (radius-es) of the centers (radial neurons)
3. LMS for finding the weights of output layer

## Plot results

Final result approximation
    gnuplot> plot 'out\plot_test_expected.txt', 'out\plot_test_calculated.txt'
    
Clustering of centers (from point 1. of the Algorithm)
    gnuplot> plot 'out\plot_clusters_train.txt', 'out\plot_clusters_init.txt', 'out\plot_clusters_out.txt
