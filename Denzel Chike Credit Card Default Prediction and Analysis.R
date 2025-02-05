# Load required libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(randomForest)
library(rpart)
library(corrplot)
library(scales)
library(gridExtra)
library(dplyr)

install.packages("ggplot2") 
install.packages("dplyr") 
install.packages("caret") 
install.packages("tidyverse") 
install.packages("randomForest") 
install.packages("rpart") 
install.packages("corrplot") 
install.packages("stats")


# Read the data
credit_card <- read.csv("C:/Users/Chike/Downloads/UCI_Credit_Card(in).csv")






# Create features
credit_card$credit_utilization <- ifelse(credit_card$LIMIT_BAL > 0, credit_card$BILL_AMT1 / credit_card$LIMIT_BAL, NA)
credit_card$utilization_category <- cut(credit_card$credit_utilization, 
                                        breaks = c(-Inf, 0.3, 0.6, 1, Inf),
                                        labels = c("Low (<30%)", "Medium (30-60%)", 
                                                   "High (60-100%)", "Very High (>100%)"))
credit_card$payment_history <- factor(ifelse(credit_card$PAY_0 <= 0, "Current", 
                                             ifelse(credit_card$PAY_0 <= 2, "1-2 Months Late", "3+ Months Late")))






# Statistical Summary

print("Overall Statistical Summary:")
summary_stats <- summary(credit_card[c("LIMIT_BAL", "AGE", "credit_utilization", 
                                       "BILL_AMT1", "PAY_AMT1", "default.payment.next.month")])
print(summary_stats)

# Distribution of default
print("Default Distribution:")
print(table(credit_card$default.payment.next.month))

# Correlation matrix of numeric variables
numeric_vars <- credit_card[c("LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "credit_utilization")]
correlation_matrix <- cor(numeric_vars, use = "complete.obs")
print("Correlation Matrix:")
print(round(correlation_matrix, 2))









# Question 1: How does the probability of default payment vary by demographics?
p1 <- ggplot(credit_card, aes(x = as.factor(EDUCATION), 
                              fill = as.factor(default.payment.next.month))) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Education Level",
       x = "Education Level", y = "Proportion",
       fill = "Default") +
  theme_minimal()

print(p1)

# Build a logistic regression model for demographics
demo_model <- glm(default.payment.next.month ~ EDUCATION + AGE + SEX + MARRIAGE,
                  family = binomial(link = "logit"),
                  data = credit_card)
print("Demographic Logistic Regression Summary:")
print(summary(demo_model))

#Visualization: The bar chart shows the proportion of defaults across different education levels. Higher education levels (e.g., university) have lower default rates compared to lower education levels.
#Model: The logistic regression indicates that education, sex, and marital status significantly affect the probability of default. For example:
  #Education: Higher education levels are associated with a lower likelihood of default.
  #Sex: Males are more likely to default than females.
  #Marriage: Married individuals are less likely to default compared to others.







# Question 2: What are the most important variables in predicting default?
set.seed(123)
features <- c("LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1", "credit_utilization")
train_index <- createDataPartition(credit_card$default.payment.next.month, p = 0.7, list = FALSE)
train_data <- credit_card[train_index,]
test_data <- credit_card[-train_index,]

# Train Random Forest
model_rf <- randomForest(x = train_data[, features], 
                         y = as.factor(train_data$default.payment.next.month),
                         ntree = 100, 
                         importance = TRUE)

# Model performance
predictions <- predict(model_rf, test_data[, features])
conf_matrix <- confusionMatrix(predictions, as.factor(test_data$default.payment.next.month))
print("Random Forest Model Performance:")
print(conf_matrix)

# Variable importance plot
importance_df <- data.frame(
  Variable = rownames(importance(model_rf)),
  Importance = importance(model_rf)[,3]
) %>%
  arrange(desc(Importance))

p2 <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance from Random Forest",
       x = "Variables",
       y = "Importance Score") +
  theme_minimal()

print(p2)

#Visualization: The Random Forest variable importance plot highlights that PAY_0 (payment status), credit_utilization, and LIMIT_BAL (credit limit) are the most important predictors of default.
#Model Performance: The Random Forest model achieved an accuracy of 80.89%, with high sensitivity (94.31%) but low specificity (32.60%). This means the model is good at identifying non-defaulters but struggles with defaulters.








# Question 3: Relationship between credit utilization, payment history, and default
p3 <- ggplot(credit_card, aes(x = utilization_category, 
                              fill = as.factor(default.payment.next.month))) +
  geom_bar(position = "fill") +
  facet_wrap(~payment_history) +
  labs(title = "Default Rate by Credit Utilization and Payment History",
       x = "Credit Utilization",
       y = "Proportion",
       fill = "Default") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)


# Build a logistic regression model for credit behavior
credit_model <- glm(default.payment.next.month ~ credit_utilization + payment_history,
                    family = binomial(link = "logit"),
                    data = credit_card)
print("Credit Behavior Logistic Regression Summary:")
print(summary(credit_model))


#Visualization: The bar chart shows that higher credit utilization and worse payment history (e.g., "3+ Months Late") are associated with higher default rates.
#Model: The logistic regression confirms that:
  #Credit Utilization: Higher utilization significantly increases the likelihood of default.
  #Payment History: Being "3+ Months Late" drastically increases the probability of default, while being "Current" reduces it.