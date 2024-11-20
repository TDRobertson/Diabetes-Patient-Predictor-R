# Load necessary libraries
if (!requireNamespace("mice", quietly = TRUE)) install.packages("mice")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("tidymodels", quietly = TRUE)) install.packages("tidymodels")
if (!requireNamespace("rpart.plot", quietly = TRUE)) install.packages("rpart.plot")

library(rpart.plot)
library(tidyverse)
library(tidymodels)
library(mice)
library(caret)
library(corrplot)
library(ggplot2)

# Classification tree written by Kristian Obrusanszki
# Define recipe, outcome is reliant on all the other variables
completed_data$Outcome <- factor(completed_data$Outcome, levels = c(0, 1))
rec <- recipe(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age, 
              data = completed_data)

set.seed(14092022)

# Classification decision tree
tree <- decision_tree(
  mode = "classification"
) %>% set_engine("rpart")      #specify the engine if not set already

# Create a workflow and add recipe and model to it
tree_wflow <- workflow() |>
  add_recipe(rec) |>
  add_model(tree)

# Fit the model to the data
fit_tree <- fit(tree_wflow, data = completed_data)

# Print the fit object to inspect it
print(fit_tree)

# Prepare data for prediction by pre-processing with the recipe
pred <- rec |>
  prep() |>
  bake(new_data = NULL)

# Use `augment()` to add predictions to the dataset
data_aug <- augment(fit_tree, pred)

# Preview the augmented data with predictions
head(data_aug)

# Calculate accuracy: specify Outcome as truth and .pred_class as estimate
accuracy_result <- data_aug %>%
  yardstick::accuracy(truth = Outcome, estimate = .pred_class)
sensitivity_result <- data_aug %>%
  yardstick::sensitivity(truth = Outcome, estimate = .pred_class)
specificity_result <- data_aug %>%
  yardstick::specificity(truth = Outcome, estimate = .pred_class)

# Print accuracy
print(accuracy_result)
print(sensitivity_result)
print(specificity_result)

# Add the original data for reference
predictions <- predict(fit_tree, completed_data, type = "prob") %>%
  bind_cols(completed_data) 

# Model made opposite predictions so we just FLIP EM
predictions <- predictions %>%
  mutate(.pred_1 = 1 - .pred_1)

# Calculate ROC curve
roc_data <- roc_curve(predictions, truth = Outcome, .pred_1) # Assuming "1" is the positive class

# Plot ROC curve
autoplot(roc_data) +
  ggtitle("ROC Curve for Classification Tree Model") +
  theme_minimal()
