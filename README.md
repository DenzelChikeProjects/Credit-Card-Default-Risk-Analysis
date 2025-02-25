# Credit-Card-Default-Risk-Analysis
---
title: "Credit Card Default Risk Analysis"
author: "Denzel Chike"
date: "2025-01-10"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## by Denzel Chike

### Introduction

Understanding the factors that influence credit card default risk is essential for financial institutions to mitigate losses and for individuals to manage their credit responsibly. This project focuses on analyzing credit card default risk using a dataset from UCI. By leveraging statistical analysis and machine learning techniques, the project aims to answer key analytical questions:


-   How does the probability of default payment vary by demographics? 

-   What are the most important variables in predicting default? 

-   Relationship between credit utilization, payment history, and default?


This project aims to enhance credit risk assessment models and improve decision-making processes in financial institutions. 

#Denzel Chike Project over Credit Card Default Risk


**#Libraries**

```{r}
library(tidyverse)
library(ggplot2)
library(caret)
library(randomForest)
library(rpart)
library(corrplot)
library(scales)
library(gridExtra)
library(dplyr)
```

install.packages("tidyverse")
install.packages("ggplot2")
install.packages("caret")
install.packages("randomForest")
install.packages("rpart")
install.packages("corrplot")
install.packages("scales")
install.packages("gridExtra")
install.packages("dplyr")

**#Data Loaded**

```{r}
credit_card <- read.csv("C:/Users/Chike/Downloads/UCI_Credit_Card(in).csv")
```

**#Create Features**

```{r}
credit_card$credit_utilization <- ifelse(credit_card$LIMIT_BAL > 0, credit_card$BILL_AMT1 / credit_card$LIMIT_BAL, NA)
credit_card$utilization_category <- cut(credit_card$credit_utilization,
										breaks = c(-Inf, 0.3, 0.6, 1, Inf),
										labels = c("Low (<30%)", "Medium (30-60%)",
													"High (60-100%)", "Very High (>100%)"))
credit_card$payment_history <- factor(ifelse(credit_card$PAY_0 <= 0, "Current",
												ifelse(credit_card$PAY_0 <= 2, "1-2 Months Late", "3+ Months Late")))
```

**#Statistical Summary**

```{r}
print("Overall Statistical Summary:")
summary_stats <- summary(credit_card[c("LIMIT_BAL", "AGE", "credit_utilization",
										"BILL_AMT1", "PAY_AMT1", "default.payment.next.month")])
print(summary_stats)

```

**#Distribution of default**

```{r}
print("Default Distribution:")
print(table(credit_card$default.payment.next.month))
```

**#Correlation matrix of numeric variables**

```{r}
numeric_vars <- credit_card[c("LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "credit_utilization")]
correlation_matrix <- cor(numeric_vars, use = "complete.obs")
print("Correlation Matrix:")
print(round(correlation_matrix, 2))
```

**#Question 1: How does the probability of default payment vary by demographics?** 

```{r}
p1 <- ggplot(credit_card, aes(x = as.factor(EDUCATION),
								              fill = as.factor(default.payment.next.month))) +
	geom_bar(position = "fill") +
	labs(title = "Default Rate by Education Level",
	     x = "Education Level", y = "Proportion",
	     fill = "Default") +
	theme_minimal()
		
print(p1)

#Logistic Regression model for demographics
demo_model <- glm(default.payment.next.month ~ EDUCATION + AGE + SEX + MARRIAGE,
					family = binomial(link = "logit"),
					data = credit_card)
print("Demographic Logistic Regreession Summary:")
print(summary(demo_model))


```

**#Question 2: What are the most important variables in predicting default?** 

```{r}
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

```

**#Question 3: Relationship between credit utilization, payment history, and default?**

```{r}
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

# Logistic regression model for credit behavior 	
credit_model <- glm(default.payment.next.month ~ credit_utilization + payment_history,
					      family = binomial(link = "logit"),
					      data = credit_card)

print("Credit Behavior Logistic Regression Summary:")
print(summary(credit_model))

```

## Conclusion

This analysis provides valuable insights into credit card default risk. Demographic factors such as education, age, sex, and marital status influence the likelihood of default. The Random Forest model identified key predictors like payment status, credit utilization, and bill amounts, highlighting their importance in credit risk assessment. Additionally, higher credit utilization and poor payment history are strongly associated with increased default rates. These findings can assist financial institutions in developing robust credit scoring models and implementing effective risk management strategies.


## References

https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?select=UCI_Credit_Card.csv
