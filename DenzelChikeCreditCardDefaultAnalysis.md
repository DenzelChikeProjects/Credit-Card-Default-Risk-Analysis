Credit Card Default Risk Analysis
================
Denzel Chike
2025-01-10

## by Denzel Chike

### Introduction

Understanding the factors that influence credit card default risk is
essential for financial institutions to mitigate losses and for
individuals to manage their credit responsibly. This project focuses on
analyzing credit card default risk using a dataset from UCI. By
leveraging statistical analysis and machine learning techniques, the
project aims to answer key analytical questions:

- How does the probability of default payment vary by demographics?

- What are the most important variables in predicting default?

- Relationship between credit utilization, payment history, and default?

This project aims to enhance credit risk assessment models and improve
decision-making processes in financial institutions.

\#Denzel Chike Project over Credit Card Default Risk

**\#Libraries**

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(ggplot2)
library(caret)
```

    ## Loading required package: lattice
    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(randomForest)
```

    ## randomForest 4.7-1.2
    ## Type rfNews() to see new features/changes/bug fixes.
    ## 
    ## Attaching package: 'randomForest'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine
    ## 
    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(rpart)
library(corrplot)
```

    ## corrplot 0.95 loaded

``` r
library(scales)
```

    ## 
    ## Attaching package: 'scales'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     discard
    ## 
    ## The following object is masked from 'package:readr':
    ## 
    ##     col_factor

``` r
library(gridExtra)
```

    ## 
    ## Attaching package: 'gridExtra'
    ## 
    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
library(dplyr)
```

install.packages(“tidyverse”) install.packages(“ggplot2”)
install.packages(“caret”) install.packages(“randomForest”)
install.packages(“rpart”) install.packages(“corrplot”)
install.packages(“scales”) install.packages(“gridExtra”)
install.packages(“dplyr”)

**\#Data Loaded**

``` r
credit_card <- read.csv("C:/Users/Chike/Downloads/UCI_Credit_Card(in).csv")
```

**\#Create Features**

``` r
credit_card$credit_utilization <- ifelse(credit_card$LIMIT_BAL > 0, credit_card$BILL_AMT1 / credit_card$LIMIT_BAL, NA)
credit_card$utilization_category <- cut(credit_card$credit_utilization,
                                        breaks = c(-Inf, 0.3, 0.6, 1, Inf),
                                        labels = c("Low (<30%)", "Medium (30-60%)",
                                                    "High (60-100%)", "Very High (>100%)"))
credit_card$payment_history <- factor(ifelse(credit_card$PAY_0 <= 0, "Current",
                                                ifelse(credit_card$PAY_0 <= 2, "1-2 Months Late", "3+ Months Late")))
```

**\#Statistical Summary**

``` r
print("Overall Statistical Summary:")
```

    ## [1] "Overall Statistical Summary:"

``` r
summary_stats <- summary(credit_card[c("LIMIT_BAL", "AGE", "credit_utilization",
                                        "BILL_AMT1", "PAY_AMT1", "default.payment.next.month")])
print(summary_stats)
```

    ##    LIMIT_BAL            AGE        credit_utilization   BILL_AMT1      
    ##  Min.   :  10000   Min.   :21.00   Min.   :-0.61989   Min.   :-165580  
    ##  1st Qu.:  50000   1st Qu.:28.00   1st Qu.: 0.02203   1st Qu.:   3559  
    ##  Median : 140000   Median :34.00   Median : 0.31399   Median :  22382  
    ##  Mean   : 167484   Mean   :35.49   Mean   : 0.42377   Mean   :  51223  
    ##  3rd Qu.: 240000   3rd Qu.:41.00   3rd Qu.: 0.82984   3rd Qu.:  67091  
    ##  Max.   :1000000   Max.   :79.00   Max.   : 6.45530   Max.   : 964511  
    ##     PAY_AMT1      default.payment.next.month
    ##  Min.   :     0   Min.   :0.0000            
    ##  1st Qu.:  1000   1st Qu.:0.0000            
    ##  Median :  2100   Median :0.0000            
    ##  Mean   :  5664   Mean   :0.2212            
    ##  3rd Qu.:  5006   3rd Qu.:0.0000            
    ##  Max.   :873552   Max.   :1.0000

**\#Distribution of default**

``` r
print("Default Distribution:")
```

    ## [1] "Default Distribution:"

``` r
print(table(credit_card$default.payment.next.month))
```

    ## 
    ##     0     1 
    ## 23364  6636

**\#Correlation matrix of numeric variables**

``` r
numeric_vars <- credit_card[c("LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "credit_utilization")]
correlation_matrix <- cor(numeric_vars, use = "complete.obs")
print("Correlation Matrix:")
```

    ## [1] "Correlation Matrix:"

``` r
print(round(correlation_matrix, 2))
```

    ##                    LIMIT_BAL   AGE BILL_AMT1 PAY_AMT1 credit_utilization
    ## LIMIT_BAL               1.00  0.14      0.29     0.20              -0.37
    ## AGE                     0.14  1.00      0.06     0.03              -0.03
    ## BILL_AMT1               0.29  0.06      1.00     0.14               0.57
    ## PAY_AMT1                0.20  0.03      0.14     1.00              -0.02
    ## credit_utilization     -0.37 -0.03      0.57    -0.02               1.00

**\#Question 1: How does the probability of default payment vary by
demographics?**

``` r
p1 <- ggplot(credit_card, aes(x = as.factor(EDUCATION),
                                              fill = as.factor(default.payment.next.month))) +
    geom_bar(position = "fill") +
    labs(title = "Default Rate by Education Level",
         x = "Education Level", y = "Proportion",
         fill = "Default") +
    theme_minimal()
        
print(p1)
```

![](DenzelChikeCreditCardDefaultAnalysis_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
#Logistic Regression model for demographics
demo_model <- glm(default.payment.next.month ~ EDUCATION + AGE + SEX + MARRIAGE,
                    family = binomial(link = "logit"),
                    data = credit_card)
print("Demographic Logistic Regreession Summary:")
```

    ## [1] "Demographic Logistic Regreession Summary:"

``` r
print(summary(demo_model))
```

    ## 
    ## Call:
    ## glm(formula = default.payment.next.month ~ EDUCATION + AGE + 
    ##     SEX + MARRIAGE, family = binomial(link = "logit"), data = credit_card)
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -0.872667   0.108472  -8.045 8.62e-16 ***
    ## EDUCATION    0.077997   0.017675   4.413 1.02e-05 ***
    ## AGE         -0.001111   0.001681  -0.661 0.508889    
    ## SEX         -0.202746   0.028424  -7.133 9.82e-13 ***
    ## MARRIAGE    -0.109947   0.029530  -3.723 0.000197 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 31705  on 29999  degrees of freedom
    ## Residual deviance: 31619  on 29995  degrees of freedom
    ## AIC: 31629
    ## 
    ## Number of Fisher Scoring iterations: 4

**\#Question 2: What are the most important variables in predicting
default?**

``` r
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
```

    ## [1] "Random Forest Model Performance:"

``` r
print(conf_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 6642 1319
    ##          1  401  638
    ##                                          
    ##                Accuracy : 0.8089         
    ##                  95% CI : (0.8006, 0.817)
    ##     No Information Rate : 0.7826         
    ##     P-Value [Acc > NIR] : 4.275e-10      
    ##                                          
    ##                   Kappa : 0.3239         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.9431         
    ##             Specificity : 0.3260         
    ##          Pos Pred Value : 0.8343         
    ##          Neg Pred Value : 0.6141         
    ##              Prevalence : 0.7826         
    ##          Detection Rate : 0.7380         
    ##    Detection Prevalence : 0.8846         
    ##       Balanced Accuracy : 0.6345         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

``` r
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

![](DenzelChikeCreditCardDefaultAnalysis_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

**\#Question 3: Relationship between credit utilization, payment
history, and default?**

``` r
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
```

![](DenzelChikeCreditCardDefaultAnalysis_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
# Logistic regression model for credit behavior     
credit_model <- glm(default.payment.next.month ~ credit_utilization + payment_history,
                          family = binomial(link = "logit"),
                          data = credit_card)

print("Credit Behavior Logistic Regression Summary:")
```

    ## [1] "Credit Behavior Logistic Regression Summary:"

``` r
print(summary(credit_model))
```

    ## 
    ## Call:
    ## glm(formula = default.payment.next.month ~ credit_utilization + 
    ##     payment_history, family = binomial(link = "logit"), data = credit_card)
    ## 
    ## Coefficients:
    ##                               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                   -0.23199    0.02996  -7.744 9.61e-15 ***
    ## credit_utilization             0.39614    0.03555  11.143  < 2e-16 ***
    ## payment_history3+ Months Late  0.91764    0.10691   8.583  < 2e-16 ***
    ## payment_historyCurrent        -1.76919    0.03158 -56.022  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 31705  on 29999  degrees of freedom
    ## Residual deviance: 27868  on 29996  degrees of freedom
    ## AIC: 27876
    ## 
    ## Number of Fisher Scoring iterations: 4

## Conclusion

This analysis provides valuable insights into credit card default risk.
Demographic factors such as education, age, sex, and marital status
influence the likelihood of default. The Random Forest model identified
key predictors like payment status, credit utilization, and bill
amounts, highlighting their importance in credit risk assessment.
Additionally, higher credit utilization and poor payment history are
strongly associated with increased default rates. These findings can
assist financial institutions in developing robust credit scoring models
and implementing effective risk management strategies.

## References

<https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?select=UCI_Credit_Card.csv>
