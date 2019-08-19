
retrieveRelevantSets <- function(yearSets) {
  newYears <- list()
  j <- 1
  for (i in 1:ncol(yearSets)) {
    rel.col <- as.numeric(yearSets[,i])
    good.seq <- seq(rel.col[1], rel.col[1] + (nrow(yearSets)-1), by=1)
    if (all(good.seq==rel.col)) {
      newYears[[j]] <- yearSets[,i]
      j <- j + 1
    }
  }
  return(newYears)
}

retrieveDifferentLevels <- function(train, test, factor.cols) {
  
  train.test.factors <- intersect(colnames(train), factor.cols)
  
  diff.levels <- c()
  for (fac in train.test.factors) {
    train.fac.levels <- unique(train[,c(fac)])
    test.fac.levels <- unique(test[,c(fac)])
    if ((length(setdiff(test.fac.levels, train.fac.levels))!=0) & (!all(is.na(test.fac.levels)))) {
      diff.levels <- c(diff.levels, fac)
    }
  }
  return(diff.levels)
  
}

correctLevels <- function(train, test, diff.levels) {
  for (col in 1:ncol(test)) {
    
    if (colnames(test)[col] %in% diff.levels) {
      new.test.levels <- setdiff(unique(test[,col]), unique(train[,col]))
      test[test[,col] %in% new.test.levels, col] <- NA
      diff.level <- setdiff(unique(test[,col]), unique(train[,col]))
      stopifnot(length(diff.level)==0 | is.na(diff.level))
    }
    
  }
  return(test)
  
}

plotUsedVars <- function(chosen.vars, vars.per.plot) {
  chosen.vars <- chosen.vars[order(chosen.vars$Variable),]
  num.vars <- nrow(chosen.vars)
  num.plots <- ceiling(num.vars / vars.per.plot)
  plot.indicies <- seq(vars.per.plot, num.vars, by=vars.per.plot)
  plot.indicies <- c(1, plot.indicies)
  #print(plot.indicies)
  plot.list <- list()
  i <- 1
  for (plot.num in plot.indicies) {
    start.num <- plot.num; end.num <- ifelse((start.num + vars.per.plot - 1) > num.vars, num.vars, start.num + vars.per.plot - 1); #print(chosen.vars[start.num:end.num, ])
    chosen.vars.long <- melt(chosen.vars[start.num:end.num, ], id.vars = 'Variable');# print(head(chosen.vars.long))
    chosen.vars.long$fill.colour[chosen.vars.long$value==1] <- 'green3'
    chosen.vars.long$fill.colour[chosen.vars.long$value==0.5] <- 'orange'
    chosen.vars.long$fill.colour[chosen.vars.long$value==0] <- 'red2'
    plot <- ggplot(data=chosen.vars.long, aes(x=Variable, y=variable, fill=value)) +
      geom_tile(colour = "white") +
      #scale_fill_manual(values = c("green3",  "red2", "orange")) +
      scale_fill_gradientn(colours = c("red2", "orange", "green3"), values = c(0, 0.5, 1)) +
      theme_bw() +
      ggpubr::rotate_x_text() + theme(legend.position = 'none',
                                      text = element_text(size=25)) +
      #theme(axis.text.x=element_text(angle=90, hjust = 0), legend.position="none") +
      ylab("Year")
    
    plot.list[[i]] <- plot
    i <- i + 1
  }
  return(plot.list)
}

processData <- function(data, vars, factor.cols, flag, train=NULL) {
  train.factors <- intersect(vars, factor.cols)
  train.numerics <- setdiff(vars, train.factors)
  
  if (flag=='Test') {
    for (fac in train.factors) {
      train.levels <- levels(train[,c(fac)])
      data[,c(fac)] <- factor(x=data[,c(fac)], levels = train.levels)
    }
  } else if (flag=='Train') {
    #print(head(data[,c(train.factors)]))
    data[,c(train.factors)] <- lapply(data[,train.factors] , as.factor)
  }
  
  data[,c(train.numerics)] <- lapply(data[,train.numerics], as.numeric); #print(str(train.data));
  return(data)
}

calculateAUROC <- function(predictions, target, image.path) {
  
  png(paste0(image.path, "AUROC.png"), width = 480, height = 480, type="cairo")
  plot(roc(predictions, target))
  dev.off()
  
  auc <- AUC(predictions, target)
  return(auc)
}

calculateAUPRC <- function(predictions, target, image.path) {
  
  png(paste0(image.path, "AUPRC.png"), width = 480, height = 480, type="cairo")
  PRcurve(predictions, target)
  dev.off()
  
  prauc <- PRAUC(predictions, target)
  return(prauc)
}

calculateStats <- function(predictions, target, image.path) {
  print("Calculating Stats")

  auroc <- ifelse(sum(!is.na(predictions))>1, 
                  calculateAUROC(predictions, target,
                                 image.path),
                  NA)
  
  auprc <- ifelse(sum(!is.na(predictions))>1, 
                  calculateAUPRC(predictions, target,
                                 image.path),
                  NA)
  
  fac.predictions <- factor(ifelse(predictions>0.5, 1, 0), levels=c(0, 1))
  
  PPV <- ppv_vec(truth=target, estimate=fac.predictions, prevalence=(sum(target==1)/length(target)))
  NPV <- npv_vec(truth=target, estimate=fac.predictions, prevalence=(sum(target==1)/length(target)))
  
  print(paste('AUROC:', round(auroc, 3)))
  print(paste('AUPRC:', round(auprc, 3)))
  print(paste('PPV:', round(PPV, 3)))
  print(paste('NPV:', round(NPV, 3)))
  
  return(list(auroc, auprc, PPV, NPV))
}


runLogisticClassification <- function(train, test, file.path) {
  #print(str(train)); print(str(test))
  dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
  model.file <- paste0(file.path, "model.rds"); print(paste("Saving LogReg model to:", model.file))
  print("Fit Logistic Regression")
  model <- glm(WillReturn ~.,family=binomial(link='logit'), data=train)
  sink(paste0(file.path, "summary_model.txt"))
  print(summary(model))
  sink()
  
  #saveRDS(model, file=model.file)

  print("Predict with Logistic Regression")
  fitted.results <- predict(model, test, type='response')

  stats <- calculateStats(fitted.results, test$WillReturn, file.path)

  return(stats)
}

runGlnnetClassification <- function(train, test, file.path) {
  #print(str(train)); print(str(test))
  dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
  model.file <- paste0(file.path, "model.rds"); print(paste("Saving glmnet model to:", model.file))
  print("Fit glmnet Regression")
  y <- train[,c("WillReturn")]
  factors <- names(lapply(train, is.factor)[lapply(train, is.factor)==TRUE])
  factors <- factors[!factors %in% c("WillReturn")]
  xfactors <- model.matrix(y ~ ., data=train[,c(factors)])[, -1]
  x <- as.matrix(data.frame(train[,!colnames(train) %in% c(factors, "WillReturn")], xfactors))
  set.seed(123) 
  cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial", nfolds = 5)
  png(paste0(file.path, "cv_lasso.png"), width=480, height=480, type="cairo")
  plot(cv.lasso)
  dev.off()
  print(paste("Optimal Lambda", cv.lasso$lambda.min))
  model <- glmnet(x, y, alpha = 1, family = "binomial",
                  lambda = cv.lasso$lambda.min)
  sink(paste0(file.path, "summary_model.txt"))
  print(model)
  sink()
  
  #saveRDS(model, file=model.file)
  y.test <- test[,c("WillReturn")]
  x.testfactors <- model.matrix(y.test ~ ., data=test[,c(factors)])[, -1]
  x.test <- as.matrix(data.frame(test[,!colnames(test) %in% c(factors, "WillReturn")], x.testfactors))
  #x.test <- model.matrix(WillReturn ~ ., data=test); print(dim(x.test)); print(dim(test))
  print("Predict with glmnet Regression")
  fitted.results <- predict(model, x.test, type='response')
  
  stats <- calculateStats(fitted.results, test$WillReturn, file.path)
  
  return(stats)
}

runDecisionTreeClassification <- function(train, test, file.path, pruned=FALSE) {
  dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
  model.file <- paste0(file.path, "model.rds"); print(paste("Saving tree model to:", model.file))
  print("Fit Tree Classification")
  model <- rpart(WillReturn ~ ., data=train, method="class", 
                 control =rpart.control(minsplit =1,minbucket=1, cp=0))
  #saveRDS(model, file=model.file)
  
  png(paste0(file.path, "tree_results.png"), width = 900, height = 900, type="cairo")
  summary(model)
  dev.off()
  
  png(paste0(file.path, "tree_CVresults.png"), width = 900, height = 900, type="cairo")
  plotcp(model)
  dev.off()
  
  png(paste0(file.path, "tree.png"), width = 900, height = 900, type="cairo")
  plot(model)
  dev.off()
  
  if (pruned) {
    #print(head(model$cptable))
    rel.table.values <- model$cptabale[model$cptable[,"nsplit"] > 0,]
    model <- prune(model, cp=rel.table.values[which.min(rel.table.values[,"xerror"]),"CP"])

    png(paste0(file.path, "tree_results_pruned.png"), width = 900, height = 900, type="cairo")
    summary(model)
    dev.off()

    png(paste0(file.path, "tree_pruned.png"), width = 900, height = 900, type="cairo")
    plot(model)
    dev.off()
    
  }


  print("Predict with Tree Model")
  prediction <- predict(model, test, type = 'prob')[,2]
  
  names(prediction) <- seq(1, length(prediction), by=1); #print(head(prediction))
  
  stats <- calculateStats(as.vector(prediction), test$WillReturn, file.path)
  
  return(stats)
}




runRandomForestClassification <- function(train, test, file.path, CV=TRUE, weighted=TRUE, undersample=FALSE) {
  dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
  print("Fit Random Forest")
  if (CV | weighted) {
    optimal.wt <- ifelse(CV, runCVRFClassification(train, file.path), 10000)
    model <- randomForest(WillReturn ~ . , data = train, na.action=na.omit, ntree=300, importance=TRUE,
                          classwt=c(10, optimal.wt))
  } else if (undersample) {
    nmin <- sum(train$WillReturn == 1)
    model <- randomForest(WillReturn ~ . , data = train, na.action=na.omit, ntree=300, importance=TRUE,
                          strata = train$WillReturn, sampsize = rep(nmin, 2))
  }
  

  
  model.file <- paste0(file.path, "model.rds"); print(paste("Saving RF model to:", model.file))
  #saveRDS(model, file=model.file)

  png(paste0(file.path, "varImportance.png"), width = 900, height = 900, type="cairo")
  varImpPlot(model)
  dev.off()
  print("Predict with Random Forest")
  prediction <- predict(model, test, type = 'prob')[,2]

  names(prediction) <- seq(1, length(prediction), by=1); #print(head(prediction))

  stats <- calculateStats(as.vector(prediction), test$WillReturn, file.path)

  return(stats)
}

runCVRFClassification <- function(train, file.path) {
  factors <- names(lapply(train, is.factor)[lapply(train, is.factor)==TRUE])
  train[,c(factors)] <- lapply(train[factors], make.names)
  train[,c(factors)] <- lapply(train[factors], factor)
  x <- train[,!colnames(train) %in% c("WillReturn")]
  y <- as.factor(train[,c("WillReturn")])
  metric <- "ROC"
  customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
  customRF$parameters <- data.frame(parameter = c("classwt", "mtry", "ntree"), class = rep("numeric", 3), 
                                    label = c("classwt", "mtry", "ntree"))
  customRF$grid <- function(x, y, len = NULL, search = "grid") {}

  customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    randomForest(x, y,  ntree=param$ntree, mtry=param$mtry, classwt=c(1, param$classwt), ...)
  }
  
  customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    predict(modelFit, newdata) }
  customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    predict(modelFit, newdata, type = "prob") }
#  customRF$sort <- function(x) x[order(x[,1]),]
#  customRF$levels <- function(x) x$classes

  # train model
  control <- trainControl(method="cv", number=5, classProbs = TRUE, 
                          summaryFunction = twoClassSummary, verboseIter = TRUE)
  max.mtry <- ceiling(sqrt(ncol(x)))

  tunegrid <- expand.grid(.classwt=c(1e+03, 1e+05, 1e+07), .mtry=c(seq(2, max.mtry, by=ceiling(max.mtry/3))), .ntree=c(50, 150, 400))
  set.seed(0)
  print("Hyperparameter Search For RF")
  custom <- train(WillReturn~., data=train, method=customRF, metric=metric, tuneGrid=tunegrid, 
                  trControl=control)
  print(custom)
  write.csv(custom$results, file=paste0(file.path, "RF_tune_data.csv"))
  plot.data <- melt(custom$results[,1:4], id.vars="classwt")
  wt_plot <- ggplot(data=plot.data, aes(x=classwt, y=value, colour=variable)) + 
    geom_line(); wt_plot
  ggsave(filename = paste0(file.path, "RF_tune.png"), plot = wt_plot, type='cairo')
  
  optimal.wt <- custom$results %>% filter(Spec >= 0.5 & Spec < 0.9)
  if (nrow(optimal.wt)==0) {
    optimal.wt <- custom$results[custom$results$ROC == max(custom$results$ROC),]$classwt
  } else {
    optimal.wt <- optimal.wt %>% filter(ROC == max(optimal.wt$ROC))
    optimal.wt <- optimal.wt$classwt
  }

  return(optimal.wt)

}

runSVMClassification <- function(train, test, file.path) {
  
  dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
  print("Fit SVM")
 
  factors <- names(lapply(train, is.factor)[lapply(train, is.factor)==TRUE])
  train[,c(factors)] <- lapply(train[factors], make.names)
  train[,c(factors)] <- lapply(train[factors], factor)
 
  grid <- expand.grid(C=c(0.05, 0.5, 1))
  ctrl <- trainControl(method="cv", number=10, classProbs=TRUE)
  set.seed(0)
  model <- train(WillReturn ~., data=train, method="svmLinear",
 			trControl=ctrl,
			metric="ROC",
			preprocess=c("center", "scale"),
			tuneGrid=grid,
			tuneLength=10,
			verbose = TRUE)

  #saveRDS(model, file=model.file)

  print("Predict with SVM")
  
  prediction <- predict(model, newdata=test, type='response') 

  stats <- calculateStats(prediction, test$WillReturn, file.path)
  
  return(stats)
}

runNaiveBayesClassification <- function(train, test, file.path) {
  
  dir.create(file.path(file.path), showWarnings = FALSE, recursive = TRUE)
  print("Fit Naive Bayes")
  model <- naiveBayes(WillReturn ~ ., train)
  model.file <- paste0(file.path, "model.rds"); print(paste("Saving Naive Bayes model to:", model.file))
  #saveRDS(model, file=model.file)
  
  print("Predict with Naive Bayes")
  
  prediction <- predict(model, test, type='raw')[,1]
  
  stats <- calculateStats(prediction, test$WillReturn, file.path)
  
  return(stats)
}

normalize <- function(x) {
  denom <- max(x) - min(x)
  if (denom==0) {
    denom=1
  }
  return ((x - min(x)) / denom)
}


runCVNeuralNetworkClassification <- function(train, file.path, factors) {
  
  #train <- sample_n(train, 1500)
  train[,c(factors)] <- lapply(train[factors], make.names)
  train[,c(factors)] <- lapply(train[factors], factor)
  
  numFolds <- trainControl(method = 'cv', number = 5, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)
  custom <- train(WillReturn ~ . , data = train, method = 'nnet', contrasts = factors, maxit=200, MaxNWts=100000, 
                  metric='ROC', trControl = numFolds, 
                  tuneGrid=expand.grid(size=c(5, 15), decay=c(1.0e-5, 0.01)))
  
  print(custom)
  write.csv(custom$results, file=paste0(file.path, "NN_tune_data.csv"))
  plot.data <- melt(custom$results[,1:4], id.vars=c("size", "decay"))
  plot.data$VarToPlot <- paste("Size:", plot.data$size, "Decay:", plot.data$decay); plot.data$VarToPlot <- factor(plot.data$VarToPlot)
  print(head(plot.data))
  wt_plot <- ggplot(data=plot.data, aes(x=VarToPlot, y=value, fill=variable)) +
    geom_bar(stat='identity', position="dodge"); wt_plot
  ggsave(filename = paste0(file.path, "NN_tune.png"), plot = wt_plot, type='cairo')
  
  optimal.params <- custom$results %>% filter(Spec >= 0.5 & Spec < 0.9)
  if (nrow(optimal.params)==0) {
    optimal.params <- custom$results[custom$results$ROC == max(custom$results$ROC),][, c("size", "decay")]
  } else {
    optimal.params <- optimal.params %>% filter(ROC == max(optimal.params$ROC))
    optimal.params <- optimal.params[,c("size", "decay")]
  }
  
  return(optimal.params)
  
}

runNeuralNetworkClassification <- function(train, test, file.path, cv=FALSE) {
  dir.create(file.path(file.path), showWarnings = FALSE, recursive=TRUE)
  print("Fit Neural Network")
  

  factors <- names(lapply(train, is.factor)[lapply(train, is.factor)==TRUE])
  
  xnumerics <- as.data.frame(lapply(train[,!colnames(train) %in% c(factors, "WillReturn")], normalize))
  train[,!colnames(train) %in% c(factors, "WillReturn")] <- xnumerics
  #train <- upSample(train[,!colnames(train) %in% c("WillReturn")], train[,c("WillReturn")], list = FALSE, yname = "WillReturn")
  
  size <- 8; decay <- 1.0e-5
  if (cv) {
    results <- runCVNeuralNetworkClassification(train, file.path, factors)
    size <- as.numeric(results[1,1]); decay <- as.numeric(results[1,2])
  
  }
  print(paste0("Size: ", size, ", Decay: ", decay))
  model <- nnet(WillReturn ~ ., data=train, contrasts = factors,
                size=size, decay=decay, maxit=150, MaxNWts=10000)

  print("Predict with Neural Network")
  
  test[,!colnames(test) %in% c(factors, "WillReturn")] <- as.data.frame(lapply(test[,!colnames(test) %in% c(factors, "WillReturn")], normalize))
  
  prediction <- predict(model, test, "raw")

  #cm <- confusionMatrix(prediction,test$WillReturn,FALSE,'roc'); print(cm)
  stats <- calculateStats(prediction, test$WillReturn, file.path)

  return (stats)
  
}



findRelTrainVars <- function(train, vars, factors) {
  vars <- vars[!vars %in% c("Arrival_Time_9")]
  
  num.uniques <- apply(train[,c(factors)], 2, function(x) length(unique(na.exclude(x))))
  num.uniques <- names(num.uniques[num.uniques==1]); #print(num.uniques)
  vars <- vars[!vars %in% num.uniques]
  
  return(vars)
}

generateYearSets <- function(min.year, max.year, years.training.data) {
  year.sets <- combn(seq(min.year, max.year-1, by=1), years.training.data)
  year.sets <- retrieveRelevantSets(year.sets); 
  return(year.sets)
}

examineImportantVariables <- function(years.training.data, model, num.vars.per.plot) {
  
  year.sets <- generateYearSets(2008, 2018, years.training.data)
  
  if (model == "LR") {
    all.years.important.factors <- data.frame(matrix(nrow=0, ncol=6))
    colnames(all.years.important.factors) <- c("Years", "Variable", "Beta", "StdError", "Z_Value", "P_Value")
    
  } else if (model == "RF") {
    all.years.important.factors <- data.frame(matrix(nrow=0, ncol=3))
    colnames(all.years.important.factors) <- c("Years", "Variable", "MeanDecreaseAccuracy")
    
  }
  
  print("Gathering Important Features From Models From All Years")
  for (year.num in 1:(length(year.sets))) {
    
    years <- year.sets[[year.num]]
    print(years)
    file.path <- paste0("./Results/RollingYears_", years.training.data, "/Years_", paste(years, collapse="_"))
    
    if (model == "LR") {
      model <- readRDS(paste0(file.path, "/LogReg/model.rds"))
      important.factors <- data.frame(summary(model)$coef[summary(model)$coef[,4] <= .05, ])
      colnames(important.factors) <- c("Beta", "StdError", "Z_Value", "P_Value")
      important.factors$Variable <- rownames(important.factors)
      important.factors$Years <- as.character(list(years))
      
    } else if (model == "RF") {
      model <- readRDS(paste0(file.path, "/RF/model.rds"))
      important.factors <- data.frame(model$importance[,3]); colnames(important.factors) <- c("MeanDecreaseAccuracy")
      important.factors$Years <- as.character(list(years))
      important.factors$Variable <- rownames(model$importance)
      
    }
    all.years.important.factors <- rbind(all.years.important.factors, important.factors)
  }
  
  print(head(all.years.important.factors))
  
  sorted.vars <- table(all.years.important.factors$Variable)
  sorted.vars <- names(sort(sorted.vars, decreasing=T))
  
  num.plots <- ceiling(length(sorted.vars)/num.vars.per.plot)
  print("Examing Important Variables")
  for (plot in 1:num.plots) {
    start.num <- (plot * num.vars.per.plot) - num.vars.per.plot + 1
    end.num <- start.num + num.vars.per.plot - 1
    
    plot.data <- all.years.important.factors %>% filter(Variable %in% sorted.vars[start.num:end.num])
    
    if (model == "LR") {
      plot.wrap <- ggplot(data=plot.data, aes(x=Years, y=Beta)) +
        geom_bar(stat="identity", position=position_dodge()) +
        geom_errorbar(aes(ymin=Beta-StdError, ymax=Beta+StdError), width=.2,
                      position=position_dodge(.9)) +
        facet_wrap(~Variable, ncol=1, nrow=4, scales = "free_y") +
        rotate_x_text(angle=45) +
        geom_hline(yintercept=0, colour="red", size=2) +
        scale_x_discrete(limits=unique(all.years.important.factors$Years))
      
      file.path <- paste0("./Results/RollingYears_", years.training.data, "/LogReg_Important_Vars/")
    } else if (model == "RF") {
      plot.wrap <- ggplot(data=plot.data, aes(x=Years, y=MeanDecreaseAccuracy)) +
        geom_bar(stat="identity", position=position_dodge()) +
        facet_wrap(~Variable, ncol=1, nrow=4, scales = "free_y") +
        rotate_x_text(angle=45) +
        scale_x_discrete(limits=unique(all.years.important.factors$Years)); plot.wrap
      file.path <- paste0("./Results/RollingYears_", years.training.data, "/RF_Important_Vars/")
    }
    
    dir.create(file.path, showWarnings = FALSE)
    ggsave(plot = plot.wrap, filename = paste0(file.path, "Plot_", plot, ".png"), type='cairo')
  }
  
}


plotStatsGraph <- function(stats, years.training.data, date.sizes, models, min.year=2008, max.year=2018) {
  
  years <- generateYearSets(min.year, max.year, years.training.data); 
  #stats$Years <- unlist(lapply(as.character(years), function(x) rep(x, 4)))
  
  sizes <- data.frame(matrix(nrow=length(years), ncol=3))
  colnames(sizes) <- c("Years", "Rows", "Cols")
  sizes$Years <- as.character(years)
  sizes[,2:3] <- data.frame(matrix(unlist(date.sizes), nrow=length(date.sizes), ncol=2, byrow=T))
  
  
  data.long <- na.omit(melt(stats, id.vars=c("Years", "Stats")))
  data.long <- left_join(x=data.long, 
                         y=sizes)
  
  data.long$variable <- as.factor(data.long$variable)
  
  gd <- data.long %>% 
    group_by(variable, Stats) %>% 
    summarise(mean_value = mean(value, na.rm=T))
  print(data.long)
  #data.long$Stats <- factor(data.long$Stats, levels=c("AUROC", "NPV", "AUPRC", "PPV"))
  stats.plot <- ggplot(data.long, aes(x=Years, y=value, size=Cols, group=variable,colour=variable)) + #colour=variable,  
    geom_point(position=position_dodge(width=0.03)) + 
    geom_line(size=0.5) + 
    theme_bw() + ylab("Statistics") + 
    ggpubr::rotate_x_text(angle=45) + 
    scale_size_continuous(range = c(1,5)) +
    facet_wrap(~Stats, scales="free_y") #+
    #geom_hline(data = gd, aes(yintercept = mean_value, colour=variable))  
  
  image.path <- paste0("./mods_", years.training.data, "_", paste(models, collapse = "_"),"stats_graph.png")
  print(image.path)
  ggsave(image.path, stats.plot, type='cairo')
  
  
  return(stats.plot)
  
}


plotSameModelStatsPlot <- function(stats1, stats2, years.training.data1,
                                   years.training.data2, date.sizes1, date.sizes2,
                                   vars.to.plot) {
  years1 <- generateYearSets(2008, 2018, years.training.data1); 
  years2 <- generateYearSets(2008, 2018, years.training.data2)
  
  x.axis.years1 <- seq(2008, (2018-years.training.data1), by=1)
  x.axis.years2 <- seq(2008, (2018-years.training.data2), by=1)
  
  new.years1 <- sort(rep(x.axis.years1, 4))
  new.years2 <- sort(rep(x.axis.years2, 4))
  
  new.stats1 <- stats1
  new.stats1$Years <- new.years1

  new.stats2 <- stats2
  new.stats2$Years <- new.years2

  sizes1 <- data.frame(matrix(nrow=length(x.axis.years1), ncol=3));
  colnames(sizes1) <- c("Years", "Rows", "Cols")
  sizes1$Years <-x.axis.years1
  sizes1[,2:3] <- data.frame(matrix(unlist(date.sizes1), nrow=length(date.sizes1), ncol=2, byrow=T))
  

  sizes2 <- data.frame(matrix(nrow=length(x.axis.years2), ncol=3))
  colnames(sizes2) <- c("Years", "Rows", "Cols")
  sizes2$Years <- x.axis.years2
  sizes2[,2:3] <- data.frame(matrix(unlist(date.sizes2), nrow=length(date.sizes2), ncol=2, byrow=T))
  
  
  data.long1 <- na.omit(melt(new.stats1, id.vars=c("Years", "Stats")))
  data.long2 <- na.omit(melt(new.stats2, id.vars=c("Years", "Stats")))
  print(head(data.long1)); print(head(sizes1))
  data.long1 <- left_join(x=data.long1, 
                         y=sizes1)
  data.long2 <- left_join(x=data.long2, 
                         y=sizes2)
  
  data.long1$variable <- as.factor(data.long1$variable)
  data.long2$variable <- as.factor(data.long2$variable)
  
  data.long1 <- data.long1 %>% filter(variable %in% c(vars.to.plot))
  data.long2 <- data.long2 %>% filter(variable %in% c(vars.to.plot))
  
  data.long1$variable <- paste0(data.long1$variable, "_", years.training.data1)
  data.long2$variable <- paste0(data.long2$variable, "_", years.training.data2)
  
  #data.long1 <- rbind(data.long1, data.long2)
  
  gd1 <- data.long1 %>% 
    group_by(variable, Stats) %>% 
    summarise(mean_value = mean(value, na.rm=T))
  print(head(data.long1)); print(head(data.long2))
  #data.long$Stats <- factor(data.long$Stats, levels=c("AUROC", "NPV", "AUPRC", "PPV"))
  stats.plot <- ggplot() + 
    geom_segment(data=data.long1, mapping=aes(x=Years, xend=Years+years.training.data1,
                                              y=value, yend=value, colour=as.factor(Years)),
                 size=0.75) +
    geom_point(data=data.long1, aes(x=((Years+Years+years.training.data1)/2), y=value, colour=as.factor(Years), size=Cols),
               position=position_dodge(width=0.03)) +
    geom_segment(data=data.long2, mapping=aes(x=Years, xend=Years+years.training.data2,
                                              y=value, yend=value, colour=as.factor(Years)),
                 linetype=2, size=0.75) +
    geom_point(data=data.long2, aes(x=((Years+Years+years.training.data2)/2), y=value, colour=as.factor(Years), size=Cols),
               position=position_dodge(width=0.03), shape=18) +
    theme_bw() + ylab("Statistics") + 
    ggpubr::rotate_x_text(angle=45) + 
    scale_size_continuous(range = c(1,5)) +
    facet_wrap(~Stats, scales="free_y") +
    scale_x_continuous(name="Years", limits=c(2007, 2019),
                       breaks = seq(2008, 2018, by=2))
    #geom_hline(data = gd, aes(yintercept = mean_value, colour=variable))  
  
  #image.path <- paste0("./mods_", paste(models, collapse = "_"),"stats_graph.png")
  #print(image.path)
  #ggsave(image.path, stats.plot, type='cairo')
  return (stats.plot)
  
}



gatherModelStats <- function(years.training.data, models) {
  
  year.sets <- generateYearSets(2008, 2018, years.training.data)
  
  stats <- data.frame(matrix(nrow=(length(year.sets)*4), ncol=(length(models)+2)))
  colnames(stats) <- c("Years", "Stats", models)
  stats$Years <- unlist(lapply(as.character(year.sets), function(x) rep(x, 4)))
  
  for (model in models) {
    file.path <- paste0("Years_", years.training.data, "/", model, "_stats.csv")
    model.stats <- read.csv(file.path)
    stats[,c("Years", "Stats", model)] <- model.stats[,c("Years", "Stats", model)]
  }
  
  return(stats)
  
}

gatherTrainingSizes <- function(years.training.data) {
  
  year.sets <- generateYearSets(2008, 2018, years.training.data)
  
  col.names <- make.names(unlist(as.character(year.sets)))
  
  training.sizes <- data.frame(matrix(nrow=2, ncol=length(year.sets)))
  colnames(training.sizes) <- col.names
 

  file.path <- paste0("Years_", years.training.data, "/training_sizes.csv")
  training.sizes.raw <- read.csv(file.path)
  training.sizes[,c(col.names)] <- training.sizes.raw[,c(col.names)]
  
  colnames(training.sizes) <- unlist(as.character(year.sets))
  
  return(training.sizes)
  
}

models.to.run <- c("CVNN", "CVRF", "glmnet", "LR", "NB", "tree")

p.2.stats <- gatherModelStats(2, models.to.run)
p.3.stats <- gatherModelStats(3, models.to.run)
training.sizes.2 <- gatherTrainingSizes(2)
training.sizes.3 <- gatherTrainingSizes(3)

# Compare Models
plotStatsGraph(p.2.stats, 2, training.sizes.2, models.to.run) 
plotStatsGraph(p.3.stats, 3, training.sizes.3, models.to.run) 



# Compare same model over years
LR.comparison <- plotSameModelStatsPlot(p.2.stats, p.3.stats, 
                       2, 3, training.sizes.2, 
                       training.sizes.3, "LR"); LR.comparison

glmnet.comparison <- plotSameModelStatsPlot(p.2.stats, p.3.stats,
                                        2, 3, training.sizes.2,
                                        training.sizes.3, "glmnet"); glmnet.comparison

NB.comparison <- plotSameModelStatsPlot(p.2.stats, p.3.stats,
                                            2, 3, training.sizes.2,
                                            training.sizes.3, "NB"); NB.comparison

tree.comparison <- plotSameModelStatsPlot(p.2.stats, p.3.stats,
                                            2, 3, training.sizes.2,
                                            training.sizes.3, "tree"); tree.comparison

CVNN.comparison <- plotSameModelStatsPlot(p.2.stats, p.3.stats,
                                          2, 3, training.sizes.2,
                                          training.sizes.3, "CVNN"); CVNN.comparison


CVRF.comparison <- plotSameModelStatsPlot(p.2.stats, p.3.stats,
                                          2, 3, training.sizes.2,
                                          training.sizes.3, "CVRF"); CVRF.comparison
