runModels <- function(data, years.training.data, 
                                 vars.per.plot, factor.cols,
                                 models) {
  
  year.sets <- generateYearSets(min(as.numeric(data$Year)), 
                                max(as.numeric(data$Year)),
                                years.training.data)

  
  chosen.vars <- data.frame(matrix(nrow=(ncol(data)-1), ncol=(length(year.sets)+1)))
  
  chosen.vars[,1] <- colnames(data)[1:(length(colnames(data))-1)]
  colnames(chosen.vars) <- c("Variable", year.sets)
  
  # ALL MODELS ::"LR", "glmnet", "tree", "prunedtree", "CVRF", "weightedRF", "undersampledRF", "SVM", "NB", "NN"
  
  lr.results <- list()
  glmnet.results <- list()
  tree.results <- list()
  prunedtree.results <- list()
  rf.results <- list()
  weightedrf.results <- list()
  undersampledrf.results <- list()
  svm.results <- list()
  nb.results <- list()
  nn.results <- list()
  cvnn.results <- list()
  size.training.data <- list()
  
  print("================================")
  for (year.num in 1:(length(year.sets))) {
    years <- year.sets[[year.num]]
    train.data <- data %>% dplyr::filter(Year %in% c(years)); print(paste("Train Years:", paste(unique(train.data$Year), collapse = ", ")))
    test.data <- data %>% dplyr::filter(Year==(as.character(as.numeric(years[length(years)])+1))); print(paste("Test Years:", unique(test.data$Year)))
    na.col.sums <- colSums(is.na(train.data))
    empty.num <- ceiling(0.10*nrow(train.data))
    empty.variables <- names(na.col.sums[na.col.sums > empty.num]); #print(head(empty.variables))
    full.variables <- names(na.col.sums[na.col.sums <= empty.num]); #print(full.variables)
    if (!identical(empty.variables, character(0))) {
      #print(empty.variables)
      chosen.vars[chosen.vars$Variable %in% empty.variables,][,year.num+1] <- 0
      train.data <- train.data[, !names(train.data) %in% empty.variables]; 
      test.data <- test.data[, !names(test.data) %in% empty.variables];
    }
    
    print(dim(train.data))
    train.data <- train.data[complete.cases(train.data),]
    
    

    
    different.level.vars <- retrieveDifferentLevels(train.data, test.data, factor.cols)
    full.variables <- full.variables[!full.variables %in% different.level.vars]
    
    if(!identical(full.variables, character(0))) {
      chosen.vars[chosen.vars$Variable %in% full.variables,][year.num+1] <- 1
    }
    if (!is.null(different.level.vars)) {
      #print(different.level.vars)
      chosen.vars[chosen.vars$Variable %in% different.level.vars,][year.num+1] <- 0.5
      test.data <- correctLevels(train.data, test.data, different.level.vars)
    }
    
    na.test.col.sums <- colSums(is.na(test.data))
    empty.test.num <- ceiling(0.10*nrow(test.data))
    empty.test.variables <- names(na.test.col.sums[na.test.col.sums > empty.test.num]); #print(head(empty.variables))
    full.test.variables <- names(na.test.col.sums[na.test.col.sums <= empty.test.num]); #print(full.variables)
    if (!identical(empty.test.variables, character(0))) {
      #print(empty.variables)
      chosen.vars[chosen.vars$Variable %in% empty.test.variables,][,year.num+1] <- 0
      train.data <- train.data[, !names(train.data) %in% empty.test.variables]; 
      test.data <- test.data[, !names(test.data) %in% empty.test.variables];
    }
    
    test.data <- test.data[complete.cases(test.data),]
    #train.vars <- union(full.test.variables, different.level.vars)
    train.vars <- full.test.variables
    train.vars <- findRelTrainVars(train.data, train.vars, intersect(factor.cols, train.vars))
    
    train.data <- processData(train.data, train.vars, factor.cols, 'Train')

    test.data <- processData(test.data, train.vars, factor.cols, 'Test', train.data)
    
    print(paste("Training Data Dimenstions:", paste(dim(train.data), collapse = ", ")))
    print(paste("Testing Data Dimenstions:", paste(dim(test.data), collapse = ", ")))

    stopifnot("WillReturn" %in% colnames(train.data))
    stopifnot("WillReturn" %in% colnames(test.data))
    
    print(paste("Proportion WillReturn Train:", round(sum(train.data$WillReturn==1)/nrow(train.data), 3)))
    print(paste("Proportion WillReturn Test:", round(sum(test.data$WillReturn==1)/nrow(test.data), 3)))
    
    size.training.data[[year.num]] <- dim(train.data[,c(train.vars)])
    
    file.path <- paste0("./Results/RollingYears_", years.training.data, "/Years_", paste(years, collapse="_"), "/")
    print(paste("Path:", paste(file.path, collapse="_")))
    dir.create(file.path(file.path), recursive = TRUE)
    
    for (model in models) {
      if (model=="LR") {
        print("Running Logistic Classification")
        lr.results[[year.num]] <- runLogisticClassification(train.data[,c(train.vars)],
                                                            test.data[,c(train.vars)],
                                                            paste0(file.path, "LR/"))
        
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

    print("================================")
    
  }
    
  
  # plot variables
  num.plots <- ceiling(nrow(chosen.vars) / vars.per.plot);
  plot.list <- plotUsedVars(chosen.vars, vars.per.plot)
  file.path <- paste0("./Results/RollingYears_", years.training.data, "/variables_used_test.png")
  
  multi.plot <- ggarrange(plotlist = plot.list, ncol=1, nrow=num.plots, font.label=list(size=20))
  ggsave(multi.plot, filename = file.path)
  
  stats <- data.frame(matrix(nrow=(length(year.sets)*4), ncol=(length(models)+2)))
  colnames(stats) <- c("Years", "Stats", models)
  stats$Years <- unlist(lapply(as.character(year.sets), function(x) rep(x, 4)))
  stats[,2] <- c("AUROC", "AUPRC", "PPV", "NPV")

  # "LR", "glmnet", "tree", "CVRF", "weightedRF", "undersampledRF", "SVM", "NB", "NN"
  
  file.path <- paste0("./Results/RollingYears_", years.training.data, "/")
  for (model in models) {
    if (model=="LR") {
      stats$LR <- unlist(lr.results)
    } else if (model=="glmnet") {
      stats$glmnet <- unlist(glmnet.results)
    } else if (model == 'tree') {
      stats$tree <- unlist(tree.results)
    } else if (model == 'prunedtree') {
      stats$prunedtree <- unlist(prunedtree.results)
    } else if (model == "CVRF") {
      stats$RF <- unlist(rf.results)
    } else if (model=='weightedRF') {
      stats$weightedRF <- unlist(weightedrf.results)
    } else if (model=='undersampledRF') {
      stats$weightedRF <- unlist(undersampledrf.results)
    } else if (model == "SVM") {
      stats$SVM <- unlist(svm.results)
    } else if (model == "NB") {
      stats$NB <- unlist(nb.results)
    } else if (model == 'NN') {
      stats$NN <- unlist(nn.results)
    } else if (model == "CVNN") {
      stats$CVNN <- unlist(cvnn.results)
    } else {
      print("Results not being gathered")
    }
  }
  
  if (!models == "") {
    lapply(models, function(x) write.csv(x = stats[,c("Years", "Stats", x)], file = paste0("./Results/RollingYears_", years.training.data, "/", x, "_stats.csv")))
  }
  size.training.data <- as.data.frame(size.training.data)
  colnames(size.training.data) <- unlist(as.character(year.sets))
  write.csv(x = size.training.data, file = paste0("./Results/RollingYears_", years.training.data, "/training_sizes.csv"))
  
  results <- list(stats, size.training.data)
  names(results) <- c("Stats", "Size")
  
  return(results)
  
}








