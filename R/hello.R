## Hello World of Machine Learning
library(caret)

## Set working directory
setwd("~/Documents/ML/basic/hello_world_iris/R")

## Iris data from the University of California, Irvine
## Center for Machine Learning and Itelligent Systems 
## https://archive.ics.uci.edu/ml/datas/Iris
## https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

# name of local file with iris data
filename <- "../iris.csv"
data <- read.csv(filename, header=FALSE)

colnames(data) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

# create a list of 80% of the rows in the original data we can use for training
validation_index <- createDataPartition(data$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- data[-validation_index,]
# use the remaining 80% of data to training and testing the models
data <- data[validation_index,]

## Summary of the data
dim(data)
sapply(data, class)
head(data)

## list the levels for the class
levels(data$Species)

## Summarize the class distribution 
percentage <- prop.table(table(data$Species)) * 100
cbind(freq=table(data$Species), percentage=percentage)

## summarize attribute distributions
summary(data)

## split input and output
x <- data[, 1:4]
y <- data[, 5]

## boxplot 
par(mfrow=c(1, 4))
    for(i in 1:4) {
        boxplot(x[,i], main=names(iris)[i])
    }

## Barplot of class breakdown
plot(y)

## scatterplot matrix
featurePlot(x = x, y = y, plot="ellipse")
featurePlot(x = x, y = y, plot="box")

## density plots for each attribute by clas value
scales <- list(x = list(relation="free"), y = list(relation="free"))
featurePlot(x = x, y = y, plot = "density", scales = scales)

## Run algorithms using a 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

## Evaluate using 5 different algorithms:
    
# 1 Linear Discriminant Analysis (LDA)
# 2 Classification and Regression Trees (CART).
# 3 k-Nearest Neighbors (kNN).
# 4 Support Vector Machines (SVM) with a linear kernel.
# 5 Random Forest (RF)
# This is a good mixture of simple linear (LDA), nonlinear (CART, kNN) 
# and complex nonlinear methods (SVM, RF). 
# Reset the random number seed before reach run to ensure that the evaluation of 
# each algorithm is performed using exactly the same data splits.

# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=data, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=data, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=data, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=data, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=data, method="rf", metric=metric, trControl=control)

## Find the best model
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

## Compare accuracy of models
dotplot(results)

## summarize best model
print(fit.lda)

## Make predictions using the validation set
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

