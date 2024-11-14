#Author: Logan Bolton

# Correctly construct selected_data
selected_data <- as.matrix(cbind(data$Glucose, data$Insulin))

k <- 3
max_iter <- 100
n <- nrow(selected_data)
centroids <- selected_data[sample(1:n, k), , drop = FALSE]

euclidean_distance <- function(a, b) {
  sqrt(sum((a - b)^2))
}

for (iter in 1:max_iter) {
  # Assignment step
  clusters <- sapply(1:n, function(j) {
    distances <- apply(centroids, 1, function(centroid) {
      sqrt(sum((selected_data[j, ] - centroid)^2))
    })
    which.min(distances)
  })
  
  # Update step
  new_centroids_list <- lapply(1:k, function(cluster) {
    members <- selected_data[clusters == cluster, , drop = FALSE]
    if (nrow(members) > 0) {
      colMeans(members)
    } else {
      # Reinitialize centroid to a random data point
      selected_data[sample(1:n, 1), ]
    }
  })
  
  # Combine list into a matrix
  new_centroids <- do.call(rbind, new_centroids_list)
  
  # Ensure centroids and new_centroids have the same dimensions
  if (!all(dim(centroids) == dim(new_centroids))) {
    stop("Dimensions of centroids do not match.")
  }
  
  # Check for convergence
  if (all(centroids == new_centroids, na.rm = TRUE)) {
    cat("Converged at iteration:", iter, "\n")
    break
  }
  
  centroids <- new_centroids
}

# Plotting the clusters
plot(selected_data[,1], selected_data[,2], 
     col = clusters, pch = 16, 
     xlab = "Glucose", ylab = "Insulin", 
     main = "Clusters Based on Glucose and Insulin Levels")
points(centroids[,1], centroids[,2], col = 1:k, pch = 8, cex = 2)

# Analyzing cluster characteristics
clustered_data <- data.frame(selected_data, cluster = clusters)
colnames(clustered_data)[1:2] <- c("Glucose", "Insulin")  # Name the columns
cluster_means <- aggregate(. ~ cluster, data = clustered_data, FUN = mean)
print(cluster_means)
