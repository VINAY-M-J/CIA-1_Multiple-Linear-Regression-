## Load necessary libraries
library(tidyverse)
library(corrplot)
library(psych)
library(ggplot2)
library(DataExplorer)
library(car)
library(lmtest)
library(Metrics)
library(MASS)
library(glmnet)
library(dplyr)

################################################################################

## Load the dataset
setwd("C:\\Users\\Asus\\Desktop\\TRIM 4\\MLA")
getwd()
music_data=read.csv("music.csv", header=T)

################################################################################

## DATA EXPLORATION
View(music_data)
names(music_data)

# Display the first few rows of the dataset
head(music_data)
# Dimension of the dataset
dim(music_data)
# Structure of the dataset
str(music_data)

# Summary statistics of the dataset
summary(music_data)

# Check for missing values
summary(music_data)
is.na(music_data)
sum(is.na(music_data))
plot_missing(music_data)


#Understand distributions and correlations
pairs.panels(music_data)

# Correlation matrix
cor_matrix <- cor(music_data)
corrplot::corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.8)

plot_histogram(music_data)
plot_density(music_data)
plot_correlation(music_data)

################################################################################

# ASSESSING DATA QUALITY

# Check for missing values
sum(is.na(music_data))

# Check for duplicates
duplicate_rows <- music_data[duplicated(music_data), ]
nrow(duplicate_rows) # will remove the duplicates in data cleaning step

################################################################################

#DATA CLEANING

# Check for missing values
sum(is.na(music_data))

# Outlier detection using boxplots
boxplot(music_data[, -ncol(music_data)], main="Boxplot for Outlier Detection", col="orange", border="brown")

# Detecting outliers using IQR method
outliers <- apply(music_data, 2, function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR))
})
print(outliers)


# Treating outliers (example: capping outliers)
cap_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

music_data_cleaned <- as.data.frame(lapply(music_data, cap_outliers))
dim(music_data_cleaned)
music_data_cleaned

################################################################################

# MODELING

# Shuffling / mixing the dataset
dim(music_data_cleaned)
set.seed(1234)
music_mixed<-music_data_cleaned[order(runif(50)),]
training<-music_mixed[1:35,]
testing<-music_mixed[36:50,]

#Building full model
fullmodel<-lm(Popularity~.,data=training)
fullmodel
summary(fullmodel)

#Building model with relevant features
lm_relevant<-lm(Popularity~Energy + Valence + Length + Acousticness + Speechiness,data=training)
summary(lm_relevant)


# Predict and evaluate on test data
fullmodel_pred <- predict(fullmodel, newdata = testing)
fullmodel_pred
lm_relevant_pred <- predict(lm_relevant, newdata = testing)
lm_relevant_pred


# Calculate performance metrics
fullmodel_r2 <- summary(fullmodel)$r.squared
fullmodel_test_r2 <- cor(testing$Popularity, fullmodel_pred)^2


lm_relevant_r2 <- summary(lm_relevant)$r.squared
lm_relevant_test_r2 <- cor(testing$Popularity, lm_relevant_pred)^2

# Compare R-squared values
cat("Full Model - Train R2:", fullmodel_r2, "Test R2:", fullmodel_test_r2, "\n")

cat("Simplified Model - Train R2:", lm_relevant_r2, "Test R2:", lm_relevant_test_r2, "\n")

# MSE
fullmodel_mse <- mean((testing$Popularity - fullmodel_pred)^2)
cat("Full Model - MSE:", fullmodel_mse, "\n")

lm_relevant_mse <- mean((testing$Popularity - lm_relevant_pred)^2)
cat("Simplified Model  - MSE:", lm_relevant_mse, "\n")

################################################################################

# 

# Create model matrix 
X <- model.matrix(Popularity ~ ., music_data_cleaned)[, -1]
X
Y <- music_data_cleaned$Popularity
Y

# Define the lambda sequence
lambda <- 10^seq(10, -2, length = 100)
print(lambda)

# Split the data into training and validation sets
set.seed(567)
part <- sample(2, nrow(X), replace = TRUE, prob = c(0.7, 0.3))
X_train <- X[part == 1, ]
X_cv <- X[part == 2, ]
Y_train <- Y[part == 1]
Y_cv <- Y[part == 2]

################################################################################

# Perform Ridge regression
ridge_reg <- glmnet(X_train, Y_train, alpha = 0, lambda = lambda)
summary(ridge_reg)

# Find the best lambda via cross-validation
ridge_reg1 <- cv.glmnet(X_train, Y_train, alpha = 0)
bestlam <- ridge_reg1$lambda.min
print(bestlam)

# Predict on the validation set
ridge.pred <- predict(ridge_reg, s = bestlam, newx = X_cv)

# Calculate mean squared error
mse <- mean((Y_cv - ridge.pred)^2)
print(paste("Mean Squared Error:", mse))
ridge_r2 <- 1 - (sum((Y_cv - ridge.pred)^2) / sum((Y_cv - mean(Y_cv))^2))
ridge_r2

################################################################################

# Perform Lasso regression
lasso_reg <- glmnet(X_train, Y_train, alpha = 1, lambda = lambda)

# Find the best lambda via cross-validation
lasso_reg1 <- cv.glmnet(X_train, Y_train, alpha = 1)
bestlam <- lasso_reg1$lambda.min
bestlam

# Predict on the validation set
lasso.pred <- predict(lasso_reg, s = bestlam, newx = X_cv)

# Calculate mean squared error
mse <- mean((Y_cv - lasso.pred)^2)
print(paste("Mean Squared Error:", mse))

# Calculate R2 value
sst <- sum((Y_cv - mean(Y_cv))^2)
sse <- sum((Y_cv - lasso.pred)^2)
r2 <- 1 - (sse / sst)
print(paste("R²:", r2))

# Get the Lasso regression coefficients
lasso.coef <- predict(lasso_reg, type = "coefficients", s = bestlam)
print("Lasso Coefficients:")
print(lasso.coef)

##################################################################################

# Compare MSE and R² for all models

# Multiple Linear Regression (Simplified Model)
simplified_mse <- mean((testing$Popularity - lm_relevant_pred)^2)
simplified_r2 <- cor(testing$Popularity, lm_relevant_pred)^2

# Ridge Regression
ridge_mse <- mean((Y_cv - ridge.pred)^2)
ridge_r2 <- 1 - (sum((Y_cv - ridge.pred)^2) / sum((Y_cv - mean(Y_cv))^2))

# Lasso Regression
lasso_mse <- mean((Y_cv - lasso.pred)^2)
lasso_r2 <- 1 - (sum((Y_cv - lasso.pred)^2) / sum((Y_cv - mean(Y_cv))^2))

# Print performance metrics
cat("Simplified Linear Regression - MSE:", simplified_mse, "R²:", simplified_r2, "\n")
cat("Ridge Regression - MSE:", ridge_mse, "R²:", ridge_r2, "\n")
cat("Lasso Regression - MSE:", lasso_mse, "R²:", lasso_r2, "\n")

# Choose the best model based on the lowest MSE and highest R²
model_performance <- data.frame(
  Model = c("Simplified Linear Regression", "Ridge Regression", "Lasso Regression"),
  MSE = c(simplified_mse, ridge_mse, lasso_mse),
  R_squared = c(simplified_r2, ridge_r2, lasso_r2)
)

print(model_performance)

# Identify the best model based on MSE
best_model <- model_performance[which.min(model_performance$MSE), ]
cat("Best Model Based on MSE:\n")
print(best_model)

# Identify the best model based on R-squared
best_model_r2 <- model_performance[which.max(model_performance$R_squared), ]
cat("Best Model Based on R-squared:\n")
print(best_model_r2)

