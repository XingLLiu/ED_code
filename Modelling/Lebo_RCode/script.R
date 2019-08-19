library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(stringr)
library(CombMSC)
library(randomForest)
library(AUC)
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(Amelia)
require(MLmetrics)
library(DMwR)
library(yardstick)
library(e1071)
library(caret)
library(glmnet)
library(rpart)
library(nnet)
source("helper_functions.R")


#path <- "./data/EPIC_DATA/"
path <- "./"

train.num <- ceiling(nrow(EPIC)*0.75)
train <- EPIC[1:train.num,]
#train <- data.table(upSample(train, factor(train$WillReturn)))
#train$Class <- NULL
test <- EPIC[train.num:nrow(EPIC),]
sum(train$WillReturn==1)/nrow(train)
sum(test$WillReturn==1)/nrow(test)

train[train ==""] <- NA
test[test == ""] <- NA
na.col.sums <- colSums(is.na(train))
empty.num <- ceiling(0.10*nrow(train))
empty.variables <- names(na.col.sums[na.col.sums > empty.num]); print(head(empty.variables))
full.variables <- names(na.col.sums[na.col.sums <= empty.num]); print(full.variables)
if (!identical(empty.variables, character(0))) {
  #print(empty.variables)
  train <- train[, !names(train) %in% empty.variables, with=FALSE]; 
  test <- test[, !names(test) %in% empty.variables, with=FALSE];
}

print(dim(train)); dim(test)
train <- train[complete.cases(train),]

different.level.vars <- retrieveDifferentLevels(train, test, factor.columns)
full.variables <- full.variables[!full.variables %in% different.level.vars]

#########################33

if (!is.null(different.level.vars)) {
  #print(different.level.vars)
  test <- correctLevels(train, test, different.level.vars)
}

na.test.col.sums <- colSums(is.na(test))
empty.test.num <- ceiling(0.10*nrow(test))
empty.test.variables <- names(na.test.col.sums[na.test.col.sums > empty.test.num]); print(head(empty.variables))
full.test.variables <- names(na.test.col.sums[na.test.col.sums <= empty.test.num]); print(full.variables)
if (!identical(empty.test.variables, character(0))) {
  #print(empty.variables)
 
  train <- train[, !names(train) %in% empty.test.variables, with=FALSE]; 
  test <- test[, !names(test) %in% empty.test.variables, with=FALSE];
}

test <- test[complete.cases(test),]
#train.vars <- union(full.test.variables, different.level.vars)
# train.vars <- full.test.variables
# train.vars <- findRelTrainVars(train, train, intersect(factor.columns, train.vars))

# train <- processData(train, train, factor.columns, 'Train')
# 
# test <- processData(test, train, factor.columns, 'Test', train)

print(paste("Training Data Dimenstions:", paste(dim(train), collapse = ", ")))
print(paste("Testing Data Dimenstions:", paste(dim(test), collapse = ", ")))

stopifnot("WillReturn" %in% colnames(train))
stopifnot("WillReturn" %in% colnames(test))

print(paste("Proportion WillReturn Train:", round(sum(train$WillReturn==1)/nrow(train), 3)))
print(paste("Proportion WillReturn Test:", round(sum(test$WillReturn==1)/nrow(test), 3)))


file.path <- paste0("./Results/EPIC/")
print(paste("Path:", paste(file.path, collapse="_")))
dir.create(file.path(file.path), recursive = TRUE)

train$ArrivalMonth <- NULL
test$ArrivalMonth <- NULL
train.vars <- train.vars[!train.vars %in% "ArrivalMonth"]
train.facs <- intersect(train.vars, factor.columns)
train.nums <- intersect(train.vars, numerics)
train[,(train.facs):=lapply(.SD, as.factor),.SDcols=train.facs]
train[,(train.nums):=lapply(.SD, function(x) as.numeric(as.character(x))),.SDcols=train.nums]

test[,(train.facs):=lapply(.SD, as.factor),.SDcols=train.facs]
test[,(train.nums):=lapply(.SD, function(x) as.numeric(as.character(x))),.SDcols=train.nums]

for (model in models) {
  if (model=="LR") {
    print("Running Logistic Classification")
    # lr.results <- runLogisticClassification(train[,c(train.vars)],
    #                                                     test[,c(train.vars)],
    #                                                     paste0(file.path, "LR/"))
    
    dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
    model.file <- paste0(file.path, "LR_model.rds"); print(paste("Saving LogReg model to:", model.file))
    print("Fit Logistic Regression")
    model <- glm(WillReturn ~.,family=binomial(link='logit'), data=train)
    sink(paste0(file.path, "summary_model.txt"))
    print(summary(model))
    sink()
    
    #saveRDS(model, file=model.file)
    
    print("Predict with Logistic Regression")
    fitted.results <- predict(model, test, type='response')
    
    stats <- calculateStats(fitted.results, test$WillReturn, file.path)
    confusionMatrix(factor(ifelse(fitted.results>0.5, 1, 0), levels=c(0,1)), test$WillReturn)
    
  } else if (model == "glmnet") {
    print("Running glmnet Classification")
    glmnet.results[[year.num]] <- runGlnnetClassification(train.data[,c(train.vars)],
                                                          test.data[,c(train.vars)],
                                                          paste0(file.path, "glmnet/"))
    
  } else if (model == 'tree') {
    print("Running Decision Tree Classification")
    tree.results[[year.num]] <- runDecisionTreeClassification(train.data[,c(train.vars)],
                                                              test.data[,c(train.vars)],
                                                              paste0(file.path, "tree/"))
  } else if (model == 'prunedtree') {
    print("Running Pruned Decision Tree Classification")
    prunedtree.results[[year.num]] <- runDecisionTreeClassification(train.data[,c(train.vars)],
                                                                    test.data[,c(train.vars)],
                                                                    paste0(file.path, "prunedtree/"),
                                                                    pruned=TRUE)
    
    
  } else if (model == "CVRF") {
    print("Running CV Random Forest Classification")
    rf.results[[year.num]] <- runRandomForestClassification(train.data[,c(train.vars)],
                                                            test.data[,c(train.vars)],
                                                            paste0(file.path, "CVRF/"),
                                                            CV=TRUE)
    
  } else if (model == "weightedRF") {
    print("Running Weighted Random Forest Classification")
    weightedrf.results[[year.num]] <- runRandomForestClassification(train.data[,c(train.vars)],
                                                                    test.data[,c(train.vars)],
                                                                    paste0(file.path, "weightedRF/"),
                                                                    CV=FALSE, weighted=TRUE)
    
  } else if (model == "undersampledRF") {
    print("Running Undersampled Random Forest Classification")
    undersampledrf.results[[year.num]] <- runRandomForestClassification(train.data[,c(train.vars)],
                                                                        test.data[,c(train.vars)],
                                                                        paste0(file.path, "undersampledRF/"),
                                                                        CV=FALSE, weighted=FALSE,
                                                                        undersample=TRUE)
    
  } else if (model == "SVM") {
    print("Running SVM")
    svm.results[[year.num]] <- runSVMClassification(train.data[,c(train.vars)],
                                                    test.data[,c(train.vars)],
                                                    paste0(file.path, "SVM/"))
  } else if (model == "NB") {
    print("Running Naive Bayes")
    nb.results[[year.num]] <- runNaiveBayesClassification(train.data[,c(train.vars)],
                                                          test.data[,c(train.vars)],
                                                          paste0(file.path, "NB/"))
  } else if (model == "NN") {
    print("Running Neural Network")
    nn.results[[year.num]] <- runNeuralNetworkClassification(train.data[,c(train.vars)],
                                                             test.data[,c(train.vars)],
                                                             paste0(file.path, "NN/"))
  } else if (model == "CVNN") {
    print("Running CV Neural Network")
    cvnn.results[[year.num]] <- runNeuralNetworkClassification(train.data[,c(train.vars)],
                                                               test.data[,c(train.vars)],
                                                               paste0(file.path, "CVNN/"), 
                                                               cv=TRUE)
  }
}

retrieveDifferentLevels <- function(train, test, factor.cols) {

  train.test.factors <- intersect(colnames(train), factor.cols)

  diff.levels <- c()
  for (fac in train.test.factors) {
    print(fac)
    #train.fac.levels <- unique()
    #test.fac.levels <- unique()
    #& (!all(is.na(test.fac.levels)))
    set.diff.levels <- setdiff(unlist(test[,c(fac), with=FALSE]), unlist(train[,c(fac), with=FALSE]))
    set.diff.levels <- set.diff.levels[!is.na(set.diff.levels)]
    print(set.diff.levels)
    if (length(set.diff.levels)!=0) {
      print(paste("Have to remove this:::", set.diff.levels))
      diff.levels <- c(diff.levels, fac)
    }
    print("===============================")
  }
  return(diff.levels)

}
# 
correctLevels <- function(train, test, diff.levels) {
  for (col in 1:ncol(test)) {
    colname <- colnames(test)[col]
    if (colname %in% diff.levels) {
      
      print(colname)
      new.test.levels <- setdiff(unlist(test[,c(colname), with=FALSE]), unlist(train[,c(colname), with=FALSE]))
      print(new.test.levels)
      contains.factor <- unlist(lapply(test[,col, with=FALSE], function(x) x %in% new.test.levels))
      test[contains.factor, col] <- NA
      diff.level <- setdiff(unlist(test[,col, with=FALSE]), unlist(train[,col, with=FALSE]))
      stopifnot(length(diff.level)==0 | is.na(diff.level))
    }

  }
  return(test)

}

#train[,(factor.columns):=lapply(.SD, function(x) factor(x, levels=unique(x))),.SDcols=factor.columns]

# diff.levels <- retrieveDifferentLevels(train, test, factor.columns)
# test <- correctLevels(train, test, diff.levels)
# test[test == ""] <- NA
# train$Admitting.Provider <- NULL
# test$Admitting.Provider <- NULL

training.factors <- intersect(colnames(train), factor.columns); print(training.factors)
train[,(training.factors):=lapply(.SD, as.factor),.SDcols=training.factors]
test[,(training.factors):=lapply(.SD, as.factor),.SDcols=training.factors]
LR.model <- glm(WillReturn ~., data=train, family=binomial(link='logit'))



#small.test <- test[complete.cases(test),]

# preds.train <- predict(LR.model, type="response")
preds.test <- predict(LR.model,
                     newdata = test,
                     type = "response")



bin.test.preds <- ifelse(preds.test > 0.5, 1, 0)
table(bin.test.preds, test$WillReturn)
library(sjPlot)
library(sjlabelled)
library(sjmisc)


p1 <- plot_summs(fit)