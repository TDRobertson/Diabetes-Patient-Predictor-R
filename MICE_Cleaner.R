# Load necessary libraries
if (!requireNamespace("mice", quietly = TRUE)) install.packages("mice")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")

library(mice)
library(caret)
library(corrplot)
library(ggplot2)

# Load the dataset
data <- read.csv(choose.files(), stringsAsFactors = FALSE)

# Replace zeros with NA for specific columns
columns_with_zeros <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
data[columns_with_zeros] <- lapply(data[columns_with_zeros], function(x) ifelse(x == 0, NA, x))

# Perform multiple imputation using MICE
set.seed(123)  # Ensures reproducibility
imputed_data <- mice(data, method = 'pmm', m = 5, maxit = 50, seed = 500)

# Check imputation summary
summary(imputed_data)

# Complete the dataset using the first imputed dataset
completed_data <- complete(imputed_data, 1)

# Validate imputations with visualizations
# Before and after comparison for Glucose
ggplot(data, aes(x = Glucose)) + 
  geom_histogram(fill = 'blue', alpha = 0.5, bins = 20) + 
  ggtitle("Glucose Distribution Before Imputation")

ggplot(completed_data, aes(x = Glucose)) + 
  geom_histogram(fill = 'green', alpha = 0.5, bins = 20) + 
  ggtitle("Glucose Distribution After Imputation")

# Check correlation matrix to ensure realistic relationships
corrplot(cor(completed_data, use = "complete.obs"), method = "circle")

# Split data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(completed_data$Outcome, p = 0.8, list = FALSE)
trainData <- completed_data[trainIndex, ]
testData <- completed_data[-trainIndex, ]

# Fit a logistic regression model
model <- glm(Outcome ~ ., data = trainData, family = "binomial")
summary(model)

# Predict on the test set
predictions <- predict(model, testData, type = "response")
testData$Predicted <- ifelse(predictions > 0.5, 1, 0)

# Evaluate model accuracy
confusionMatrix(as.factor(testData$Predicted), as.factor(testData$Outcome))

# Save the cleaned dataset if needed
write.csv(completed_data, "cleaned_diabetes.csv", row.names = FALSE)
