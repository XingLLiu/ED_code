# Date: July 31st, 2019

# Input: unprocessed registration codes (RegistrationCodes.xlsx)
# Output: 
#         - saves processed registration codes (processedRegistrationCodes.xlsx)
#         - returns preprocessed data

library(data.table)

# ================== PREPROCESSING ================== # 

processRegistrationCodes <- function(data, data.path) {
  print("Processing Dates")
  data$VisitStartDate <- convertToDate(data$VisitStartDate)
  attr(data$VisitStartDate, "tzone") <- "EST"
  data$BirthDate <- convertToDate(data$BirthDate)
  attr(data$BirthDate, "tzone") <- "EST"
  data$StartOfVisit <- convertToDateTime(data$StartOfVisit)
  attr(data$StartOfVisit, "tzone") <- "EST"
  data$EndOfVisit <- convertToDateTime(data$EndOfVisit)
  attr(data$EndOfVisit, "tzone") <- "EST"
  
  # correct_types
  print("Correcting numeric types")
  data$LengthOfStayInMinutes <- as.numeric(as.character(data$LengthOfStayInMinutes))
  data$CTASScore <- as.numeric(as.character(data$CTASScore))
  data$Holiday <- apply(data.frame(data$Holiday), 1, function(x) gsub(" \\(STATUTORY\\)", "", x))
  
  # convert factors
  factors <- c("Sex", "Month", "DayOfWeek", "Holiday",  "IsBusinessDay", "DischargeDisposition", "EmergencyPhysician",
               "CTASScore", "Acuity", "ReasonForVisit", "GeographyID", "Home", "InsuranceResidence",
               "InsuranceType", "ReasonForVisit")
  
  print("Processing Factors")
  for (fac in factors) {
    print(fac)
    eval(parse(text=paste0("data$", fac, "<- as.factor(data$", fac, ")")))
    if (fac == "DayOfWeek") {
      eval(parse(text=paste0("data$", fac, "<- factor(data$", fac, ", levels=c('Sunday', 'Monday', 'Tuesday', 
                             'Wednesday', 'Thursday', 'Friday', 'Saturday'))")))
    } else if (fac == "Month") {
      eval(parse(text=paste0("data$", fac, "<- factor(data$", fac, ", levels=c('January', 'February', 'March', 
                             'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))")))
    } else if (fac == "Holiday") {
      eval(parse(text=paste0('data$', fac, "<- factor(data$", fac, ', levels=c("NEW YEAR', "\'S DAY\"", ", 'FAMILY DAY', 'GOOD FRIDAY', 'VICTORIA DAY', 
                             'CANADA DAY', 'SIMCOE DAY', 'LABOUR DAY', 'THANKSGIVING DAY', 'CHRISTMAS DAY', 'BOXING DAY', 'NULL'))")))
    } else if (fac == "Acuity") {
      eval(parse(text=paste0("data$", fac, "<- factor(data$", fac, ", levels=c('RESUSCITATION', 'EMERGENT', 'URGENT', 
                             'SEMI-URGENT', 'NON-URGENT', 'UNKNOWN', 'NULL'))")))
    } else if (fac %in% c("DischargeDisposition", "EmergencyPhysician", 
                          "GeographyID", "Home", "InsuranceResidence", "InsuranceType",
                          "ReasonForVisit")) { # order from max to min
      eval(parse(text=paste0("x <- data %>% dplyr::group_by(", fac, ") %>% dplyr::mutate(num.occur=n())")))
      x <- x[,c(fac, "num.occur")]
      x <- data.frame(unique(x))
      x <- x[order(x$num.occur, decreasing = TRUE),]
      relevant.levels <- as.vector(x[,c(fac)])
      eval(parse(text=paste0("data$", fac, "<- factor(data$", fac, ", levels=relevant.levels)")))
    }
  
  }

  data <- data.frame(data %>% dplyr::filter(!DischargeDisposition %in% to.remove))
  print(paste("Saving data to", data.path))
  fwrite(x = data, paste0(data.path, "processedRegistrationCodes.csv"))

  return (data)
  
}

