# Load packages
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(corrplot)

#1
# Load data
data <- read.csv("C:/Users/jayak/Downloads/Bank.csv")
str(data)
summary(data)

#2
# Check class balance
table(data$y)
prop.table(table(data$y))

# Plot numeric variables
ggplot(data, aes(x=age, fill=y)) + geom_histogram(binwidth=5)

library(ggplot2)

# Job distribution
ggplot(data, aes(x=job, fill=y)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title="Subscription by Job", x="Job", y="Count")


# Age distribution
ggplot(data, aes(x=age, fill=y)) +
  geom_histogram(binwidth = 5, color="black") +
  labs(title="Age Distribution by Subscription", x="Age", y="Count")


# Correlation matrix for numeric features
num_data <- select_if(data, is.numeric)
corrplot(cor(num_data), method="number")


#3
# Convert target to factor
data$y <- as.factor(data$y)

# Encode categorical variables
data <- data %>% mutate_if(is.character, as.factor)

# Train-test split
set.seed(123)
split <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[split, ]
test_data <- data[-split, ]

# Class balance in full data
ggplot(data, aes(x = y, fill = y)) +
  geom_bar() +
  ggtitle("Class Balance - Full Dataset") +
  ylab("Count") + xlab("Term Deposit") +
  theme_minimal()


#4
model1 <- glm(y ~ ., data = train_data, family = binomial)
summary(model1)

# Predictions
pred1 <- predict(model1, test_data, type = "response")
pred1_class <- ifelse(pred1 > 0.5, "yes", "no")
cm1<-confusionMatrix(as.factor(pred1_class), test_data$y)


# Plot confusion matrix
library(caret)
fourfoldplot(cm1$table, color = c("#FF9999", "#99CCFF"),
             main = "Confusion Matrix – Logistic Regression")



#5
model2 <- randomForest(y ~ ., data = train_data, ntree = 100)
print(model2)

# Predictions
pred2 <- predict(model2, test_data)
confusionMatrix(pred2, test_data$y)

model2 <- randomForest(y ~ ., data = train_data, ntree = 100)
varImpPlot(model2, main = "Variable Importance – Random Forest")

pred2 <- predict(model2, test_data)
cm2 <- confusionMatrix(pred2, test_data$y)
fourfoldplot(cm2$table, color = c("#FFD700", "#20B2AA"),
             main = "Confusion Matrix – Random Forest")



#6
# Compare accuracy, precision, recall, F1
eval_metrics <- function(true, pred){
  cm <- confusionMatrix(pred, true)
  acc <- cm$overall['Accuracy']
  prec <- cm$byClass['Pos Pred Value']
  rec <- cm$byClass['Sensitivity']
  f1 <- 2 * (prec * rec) / (prec + rec)
  return(c(Accuracy=acc, Precision=prec, Recall=rec, F1=f1))
}

results <- rbind(
  Logistic = eval_metrics(test_data$y, as.factor(pred1_class)),
  RandomForest = eval_metrics(test_data$y, pred2)
)
print(results)


eval_metrics <- function(true, pred){
  cm <- confusionMatrix(pred, true)
  acc <- cm$overall['Accuracy']
  prec <- cm$byClass['Pos Pred Value']
  rec <- cm$byClass['Sensitivity']
  f1 <- 2 * (prec * rec) / (prec + rec)
  return(c(Accuracy=acc, Precision=prec, Recall=rec, F1=f1))
}

results <- rbind(
  Logistic = eval_metrics(test_data$y, as.factor(pred1_class)),
  RandomForest = eval_metrics(test_data$y, pred2)
)
results_df <- as.data.frame(results)
results_df$Metric <- rownames(results_df)
results_melt <- reshape2::melt(results_df, id.vars = "Metric")

ggplot(results_melt, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison",
       y = "Score", x = "Model") +
  theme_minimal()
