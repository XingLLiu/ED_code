# NEED TO CHECK

# Adds extra processed variables to processed wellSoft Data

addExtraVariables <- function(data, data.path) {
  print(dim(data))
  print("Loading extra data sets")
  
  # load geoSpatial data
  print("Loading Geospatial")
  geoSpatial <- read.csv(paste0(data.path, "Geospatial_Variables.csv"))
  geoSpatial <- geoSpatial[!duplicated(geoSpatial$Address),]
  
  # load income data
  print("Loading Income")
  income.data <- read.csv(paste0(data.path, "Postal_Codes_Median_Income.csv"))
  
  # load registration codes
  print("Loading Registration data")
  reg_codes <- read.xlsx(paste0(data.path, "processedRegistrationCodes.xlsx"))
  
  
  # load flow variables 
  print("Loading flow variables")
  flowVariables <- readRDS(paste0(data.path, "flow_var_per_patient.rds"))
  
  
  # load will return data
  print("Loading whether or patient will return")
  willReturn <- read.csv(paste0(path, "willReturn.csv"))
  
  print("Joining With GeoSpatial Data")
  
  data$fullAddress <- ifelse(is.na(data$Address_Other_44),
                             paste(data$Address_43, data$City_45, data$Prov_46, data$Postal_Code_47, sep=", "),
                             paste(data$Address_Other_44, data$Address_43, data$City_45, data$Prov_46, data$Postal_Code_47, sep=", "))
  
  data <- dplyr::left_join(x=data,
                           y=geoSpatial,
                           by=c("fullAddress"="Address"))
  
  data$X <- NULL
  print(dim(data))
  
  print("Joining With Income Data")
  data <- dplyr::left_join(x=data,
                           y=income.data[,c("PostalCode", "mean_Median_total_income")],
                           by=c("Postal_Code_47"="PostalCode"))
  data$fullAddress <- NULL
  print(dim(data))
  
  dup.pat.nums <- data[duplicated(data$Pt_Accnt_5),]
  dup.pat.nums <- dup.pat.nums[order(dup.pat.nums$Pt_Accnt_5),]
  
  print("Joining with Registration Data")
  # restrict wellsoft entries to those in registration codes database
  data <- merge(x=reg_codes[,c("RegistrationNumber", "VisitStartDate")], 
                y=data[!data$Pt_Accnt_5 %in% dup.pat.nums$Pt_Accnt_5,], 
                by.x='RegistrationNumber', by.y ='Pt_Accnt_5'); 
  print(dim(data))
  
  print("Joining with Flow Variables")
  # join data with flow variables
  data <- merge(x=data, y=flowVariables, by.x="RegistrationNumber", by.y="RegistrationNumber")
  print(dim(data))
  
  
  print("Joining with Return Data")
  data <- merge(x=data, y=willReturn, by.x="RegistrationNumber", by.y="RegistrationNumber")
  print(dim(data))
  
  
  print("Ordering Data By Visit Start Date")
  data <- data[order(data$VisitStartDate),]
  
  return(data)
}