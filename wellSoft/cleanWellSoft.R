# Date: July 31st, 2018

# Processes Well Soft data :
#     - AUTOMATICALLY SORT COLUMNS from heuristics into empty columns, all zero and dates
#     - CLEAN ERRONEOUS INPUT based on patterns noticed by inspection
#     - PROCESS DATE COLUMNS where date columns are automatically derived and assembled from inspection
#     - MANUALLY CLEAN REMAINING VARIABLES FROM INSPECTION 
#
# Sort columns into factors, natural language, blobs, numeric, dates, practically empty, entirely empty,
#                   sensitive personal health information 
#
# Input: raw_wellSoft.csv
# Output: 
#     - saves cleaned_wellSoft.csv
#     - returns cleaned wellSoft data 



cleanWellSoft <- function(data, file.path) {
  
  print(dim(data))
  
  # replace white space with NA
  print("Replacing white spaces")
  data <- data[ , lapply(.SD, function(x) stri_replace_all_regex(x, "^$|^ $|^Na$", NA))]
  
  # ================================= CLEAN ERRONEOUS INPUT ================================= #
  
  all.columns <- colnames(data)
  cat("\nProcessing errors\n")
  pattern <- "\\\005\\(\\(\\(\\(|\\\005\\.\\(\\(\\(|\\\0051\\(\\(\\(|\\\005\\,\\(\\(\\(|\\\005\\)\\(\\(\\(|\\\005\\)\\(\\(\\(|\\\005\\-\\(\\(\\(|\\\005\\=\\(\\(\\(|\\\005\\*\\(\\(\\(|\\\0058\\(\\(\\("
  #pattern <- "\\\005\(\(\(\(|\\\005\.\(\(\(|\\0051\(\(\(|\\\005\\,\(\(\(|\\\005\)\(\(\(|\\\005\)\(\(\(|\\\005\\-\(\(\(|\\\005\=\(\(\(|\\005\*\(\(\(|\\0058\(\(\("
  pattern.matched <- sapply(data[,c(all.columns), with=FALSE], function(y) any(grep(pattern,  y)))
  sum(pattern.matched=='TRUE')
  pattern.matched <- names(pattern.matched[pattern.matched=='TRUE'])
  i <- 1
  
  
  for (name in pattern.matched) {
    print(paste0(i, ": ", name))
    eval(parse(text=paste0('data$', name, ' <- gsub(pattern, "", data$', name, ')')))
    i <- i + 1
    
    
  }
  
  # ================================= MANUALLY CLEAN REMAINING VARIABLES FROM INSPECTION ================================= #
  print("Manually Cleaning Variables")
  
  
  data$Ctas_Index_11 <- gsub("N/A", "", data$Ctas_Index_11)
  
  data$T2_Priority_12 <- gsub("Team Triage", "Triage Team", data$T2_Priority_12)
  
  data$Method_Of_Arrival_Indexed_S_33 <- gsub("Xxxx", "", data$Method_Of_Arrival_Indexed_S_33)
  
  data$Method_Of_Arrival_321 <- gsub(".*Land Ambulance.*", "Land Ambulance", data$Method_Of_Arrival_321)
  data$Method_Of_Arrival_321 <- gsub("Air Ambulance.*", "Air Ambulance", data$Method_Of_Arrival_321)
  data$Method_Of_Arrival_321 <- gsub(".*Ambulatory.*", "Ambulatory", data$Method_Of_Arrival_321)
  data$Method_Of_Arrival_321 <- gsub(".*DTS.*", "DTS", data$Method_Of_Arrival_321)
  
  
  
  data$Alias_38 <- gsub("\\\0059\\(\\(\\(", "", data$Alias_38)
  
  data$Sex_41 <- gsub("m", "M", data$Sex_41)
  data$Sex_41 <- gsub("0", "", data$Sex_41)
  
  data$Hc_Issuing_Prov_70 <- gsub("XX", "", data$Hc_Issuing_Prov_70)
  
  data$Reg_Status_75 <- gsub("XXXX", "", data$Reg_Status_75)
  
  data$Permanent_Address_77 <- gsub("\\`", "", data$Permanent_Address_77)
  
  data$Relationship_93 <- gsub("Not Available", "", data$Relationship_93)
  data$Relationship_93 <- gsub("Grandmothe|Grandmom|G\\'Mother", "Grandmother", data$Relationship_93)
  data$Relationship_93 <- gsub("^Motherr$|Mother\\,|Motherwr|Motherr\\,|Motherr|Mothrr|Mother\\,|Mothewr|Mother1|Motheran|Motehr|Mothedr|Mom|Mtoher|Mother Mo|Mother\\`|Motherd|Mothe|Mtother|Mohter", "Mother", data$Relationship_93)
  data$Relationship_93 <- gsub("Dad|Fahter", "Father", data$Relationship_93)
  data$Relationship_93 <- gsub("\\`Other", "Other", data$Relationship_93)
  data$Relationship_93 <- gsub("Foster Mot", "Fostermom", data$Relationship_93)
  data$Relationship_93 <- gsub("Legal\\ Guardian", "Guardian", data$Relationship_93)
  data$Relationship_93 <- gsub("Unknown", "", data$Relationship_93)
  
  data$Emp_Status_110 <- gsub("^Grandmotherr$|Grandmnother|Maternal Gm|G\\-Mother|Grand Mother|Grandmohter|Nana|Grannie|Paternal Gm|G\\'Mother|G Mother|Granmother|Paternal Grandm|Grandma|Grandmom|Garndmother|Grandmothe", "Grandmother", data$Emp_Status_110)
  data$Emp_Status_110 <- gsub("Foster$", "Foster Parent", data$Emp_Status_110)
  data$Emp_Status_110 <- gsub("G\\'Parents|Gparents|Grandparents|Grandparen$", "Grandparent", data$Emp_Status_110)
  data$Emp_Status_110 <- gsub("Grandpa|Grand\\-Dad|Grandfathe|Granddad|Grandad", "Grandfather", data$Emp_Status_110)
  data$Emp_Status_110 <- gsub("^Grandmotherr$", "Grandmother", data$Emp_Status_110)
  data$Emp_Status_110 <- gsub("^Not Applicable$|^Not Available$|N/A", "", data$Emp_Status_110)
  
  data$Relationship_120 <- gsub("Faather|Fahter|Faher|Fatherr|Fatherf|Fther|Dad|Ather|Father\\`|Fatherq|Fathe|Fahter|Father\\,", "Father", data$Relationship_120)
  data$Relationship_120 <- gsub("C\\.C\\.A\\.S\\.|Cas Worker|Cas", "C.A.S.", data$Relationship_120)
  data$Relationship_120 <- gsub("G\\'Mother|G\\-Mother|Grandmothe|Grandmom", "Grandmother", data$Relationship_120)
  data$Relationship_120 <- gsub("Grandpa|Granddad|Grandfathe", "Grandfather", data$Relationship_120)
  data$Relationship_120 <- gsub("Unckle", "Uncle", data$Relationship_120)
  data$Relationship_120 <- gsub("Applicable|Applic", "Available", data$Relationship_120)
  data$Relationship_120 <- gsub("Step Fatherr|Stepdad|Stepfather", "Step Father", data$Relationship_120)
  data$Relationship_120 <- gsub("Grandfatherrent", "Grandfather", data$Relationship_120)
  data$Relationship_120 <- gsub("Fatherr", "Father", data$Relationship_120)
  data$Relationship_120 <- gsub("Stepmom|Stepmother", "Step Mother", data$Relationship_120)
  
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(H|h)ospital|Hospita|(H|h)osp|Urgent Care|General|Gen|general", y), "Hospital", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("^(D|d)r|(D|d)octor|(O|o)ffice|Family|Medical Centre|^[0-9]+", y), "Doctor", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(E|e)merg|ER|Emergency", y), "Emerg", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(C|c)linic", y), "Clinic", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(H|h)ealth (C|c)entre|health center", y), "Health Centre", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("Dentist|Dental", y), "Dentist", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(H|h)ome", y), "Home", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("Bloorveiw|Bloorview|bloorview|BLOORVIEW|(R|r)ehab", y), "Rehab", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(A|a)fter (H|h)our|after hour", y), "After Hours", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("(C|c)redit (V|v)alley|Creit Valley|Credit Vallley", y), "Hospital", y)))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(grepl("Danforth", y), "Doctor", y))) # Danforth Paediatrics is a family doctor
  unique.locations <- names(which(table(data$Referring_Location_148) == 1))
  data$Referring_Location_148 <- as.vector(sapply(data$Referring_Location_148, function(y) ifelse(y %in% unique.locations, "Other", y)))
  
  
  data$Admit_Service_186 <- gsub("\\\0053\\(\\(\\(|\\\0056\\(\\(\\(", "", data$Admit_Service_186)
  
  data$R_A_Override_Minutes_295 <- as.numeric(as.character(data$R_A_Override_Minutes_295))
  
  data$Ctas_326 <- gsub("\\\005(\\:|\\+|\\;)\\(\\(\\(|\\\005A\\(\\(\\(","", data$Ctas_326)
  data$Ctas_326 <- gsub("N\\/A|NA","", data$Ctas_326)
  
  data$Potential_Study_343 <- gsub("\\?rivr|RIVR\\ \\?|\\?\\ rivr\\ study|rivr\\ study", "RIVR", data$Potential_Study_343)
  data$Potential_Study_343 <- gsub("Bronch\\ Study\\?\\?|Bronchiolitis\\ study\\,\\ Richard\\ Paged|\\?bronch\\ study", "Bronchiolitis", data$Potential_Study_343)
  data$Potential_Study_343 <- gsub("flugene study", "flugene", data$Potential_Study_343)
  data$Potential_Study_343 <- gsub("WR\\-N|wr\\ n", "WR\\ N", data$Potential_Study_343)
  data$Potential_Study_343 <- gsub("^Wr$|wr", "WR", data$Potential_Study_343)
  
  data$Pt_Weight_350 <- gsub("kg", "", data$Pt_Weight_350)
  data$Pt_Weight_350 <- as.numeric(data$Pt_Weight_350)
  
  data$Triage_Intervention_354 <- gsub("\\\0057\\(\\(\\(","", data$Triage_Intervention_354)
  
  data$Isolation_363 <- gsub("\\\005(\\+|\\:|A|3)\\(\\(\\(", "", data$Isolation_363)
  
  data$Rn_Tmplt_Athr_376 <- gsub("Sick\\ Kids", "SickKids", data$Rn_Tmplt_Athr_376)
  
  data$Private_Cc_377 <- gsub("Xxxxxxx|Xxxxxxx\\;", "", data$Private_Cc_377)
  
  data$Cpso_385 <- as.factor(data$Cpso_385)
  
  data$H_P_Template_402 <- gsub("neonate", "Neonate", data$H_P_Template_402)
  data$H_P_Template_402 <- gsub("L\\,\\ ", "", data$H_P_Template_402)
  
  data$Billing_Status_431 <- gsub("\\\005(\\+|7)\\(\\(\\(|\\ \\-\\ (TRENT|KATHLEEN)", "", data$Billing_Status_431)
  data$Billing_Status_431 <- gsub("Complete", "complete", data$Billing_Status_431)
  data$Billing_Status_431 <- gsub("Billing\\ Clerk", "billing\\ clerk", data$Billing_Status_431)
  
  data$Disposition_480 <- gsub("home", "Home", data$Disposition_480)
  
  data$Condition_At_Disposition_481 <- gsub("GOod.*|.*Good.*|.*good.*|^ood|^god$|^(G|g)oo$|^g$|GOOD.*|^G$", "Good", data$Condition_At_Disposition_481)
  data$Condition_At_Disposition_481 <- gsub("Fair.*|^f$|fair.*|FAIR.*", "Fair", data$Condition_At_Disposition_481)
  data$Condition_At_Disposition_481 <- gsub(".*Stable.*|.*stable.*|Satble", "Stable", data$Condition_At_Disposition_481)
  data$Condition_At_Disposition_481 <- gsub(".*Well.*|.*well.*", "Well", data$Condition_At_Disposition_481)
  data$Condition_At_Disposition_481 <- gsub("Ok|ok", "OK", data$Condition_At_Disposition_481)
  
  data$Bed_Type_565 <- gsub("\\\0054\\(\\(\\(", "", data$Bed_Type_565)
  data$Bed_Type_565 <- gsub("yes|^y$", "Yes", data$Bed_Type_565)
  data$Bed_Type_565 <- gsub("^n$|no", "No", data$Bed_Type_565)
  
  data$Hc_Swipe_573 <- gsub("\\`\\`\\`", "", data$Hc_Swipe_573)
  data$Hc_Swipe_573 <- gsub("\\`", "", data$Hc_Swipe_573)
  data$Hc_Swipe_573 <- gsub("0", "", data$Hc_Swipe_573)
  data$Hc_Swipe_573 <- gsub(".*Swipe", "Swipe", data$Hc_Swipe_573)
  data$Hc_Swipe_573 <- gsub(".*Scan", "Scan", data$Hc_Swipe_573)
  data$Hc_Swipe_573 <- gsub("^[Swipe|Scan]", "", data$Hc_Swipe_573)
  
  data$Arrival_Fda_627 <- gsub("\\\005\\/\\(\\(\\(|\\\0057\\(\\(\\(", "", data$Arrival_Fda_627)
  
  data$Language_56 <- gsub("^Amachric$|^Amaric$|Ethopian|^Amharic/English$|Amhric|^Ahmaric$|^Ethiopian$|^Aderic/English$|^Amrrk$",
                               "Amharic", data$Language_56)
  data$Language_56 <- gsub("^Alabian$|^Albainian$|Albamian|^Albania$|Albanian|^Albaninan$|^Albanion$|^Albian$|^Albina$|^Labanian$",
                               "Albanian", data$Language_56)
  data$Language_56 <- gsub("^Armanian$|^Armanian/Russian$|Armeian|^Armenian$|Armian|^Arminaian$",
                               "Armenian", data$Language_56)
  data$Language_56 <- gsub("^Aramic$",
                               "Arabic", data$Language_56)
  data$Language_56 <- gsub("^Asirian$|^Asserion$|^Assryain$|^Assryan$|Assryial|^Assyarn$|^Assryial$|^Assyarn$|^Assyrian$",
                               "Assyrian", data$Language_56)
  data$Language_56 <- gsub("^Bangalalesh$|^Bangali$|^Bangely$|^Bangla$|Bangladash|^Bangladeshi$|^Bangladi$|^Bangli$|^Bangoli$|^Bengala$|^Bengali$|^Bangoli$|^Bengeli$|^Bengla$|^Bengli$|^Bengoli$|^Bengoly$|^Bengali/Some English$|^Bingoli$",
                               "Bengali", data$Language_56)
  data$Language_56 <- gsub("^Bosnian$|^Boznian$",
                               "Bosnian", data$Language_56)
  data$Language_56 <- gsub("^Corean$",
                               "Korean", data$Language_56)
  data$Language_56 <- gsub("^Cough/Congestion$",
                               "", data$Language_56)
  data$Language_56 <- gsub("^Creatian$|^Croatia$|^Croatian$|^Croation$|^Creatia$",
                               "Croatian", data$Language_56)
  data$Language_56 <- gsub("^Dahri$|^Dari$|^Dira$",
                               "Dari", data$Language_56)
  data$Language_56 <- gsub("^E$",
                               "English", data$Language_56)
  data$Language_56 <- gsub("^Dagalog/ And English$",
                               "Dagalog", data$Language_56)
  data$Language_56 <- gsub("^English/Portuguese$",
                               "Portuguese", data$Language_56)
  data$Language_56 <- gsub("^English/Spanish$",
                               "Spanish", data$Language_56)
  data$Language_56 <- gsub("^English/Chinese$",
                               "Chinese", data$Language_56)
  data$Language_56 <- gsub("^French/Russian$",
                               "French", data$Language_56)
  data$Language_56 <- gsub("^Gana$",
                               "Twi", data$Language_56)
  data$Language_56 <- gsub("^Gejarti$|^Jujarati$|^Gudjarti$|^Gugarat$|^Gugarati$|^Gujarati$|^Gujerati$|^Gujraki$|^Gujrathi$|^Gujrati$|^Gurati$|^Gujrathi$|^Gujrati$|^Gurati$|^Gusurati$|^Kujerati$|^Kurshrati$",
                               "Gujarati", data$Language_56)
  data$Language_56 <- gsub("^Haka$",
                               "Hakka", data$Language_56)
  data$Language_56 <- gsub("^Haraic$|^Harar$|^Harari$|^Hararic$|^Hararie$|^Harie$|^Hariha$|^Harry$",
                               "Harar", data$Language_56)
  data$Language_56 <- gsub("^Hearing Inpaired$",
                               "Hearing Impaired", data$Language_56)
  data$Language_56 <- gsub("^Hindu,Panjabi$|^Hiri$|^Sindhi$",
                               "Hindi", data$Language_56)
  data$Language_56 <- gsub("^Kinyarwa$|^Kinyarwanda$|^Kinyrwanda$",
                               "Kinyarwanda", data$Language_56)
  data$Language_56 <- gsub("^Kiswahili$|^Swaheli$|^Swahilee$",
                               "Swahili", data$Language_56)
  data$Language_56 <- gsub("^Kurdish/Farsi$",
                               "Kurdish", data$Language_56)
  data$Language_56 <- gsub("^Lebonise$",
                               "Lebonese", data$Language_56)
  data$Language_56 <- gsub("^Lithiwanian$|^Lithuania$|^Lithuaniain$|^Lithuanian$|^Lithunian$|^Lithunrian$",
                               "Lithuanian", data$Language_56)
  data$Language_56 <- gsub("^Macedonean$|^Macedonia$|^Macedonian$|^Masedonian$|^Masadonian$|^Masedonian$|^Masidonian$|^Masodonian$|^Masadoian$",
                               "Macedonian", data$Language_56)
  data$Language_56 <- gsub("^Malayalam$|^Malayalan$|^Malayam$|^Malyalam$",
                               "Malayalam", data$Language_56)
  data$Language_56 <- gsub("^Mandarin$|^Mandarin/English$|^Manderine$|^Malyalam$",
                               "Mandarin", data$Language_56)
  data$Language_56 <- gsub("^Marthi$",
                               "Marathi", data$Language_56)
  data$Language_56 <- gsub("^Mongol$|^Mongolia$|^Mongolian$",
                               "Mongolian", data$Language_56)
  data$Language_56 <- gsub("^Myanmavnese$|^Myanmer$",
                               "Burmese", data$Language_56)
  data$Language_56 <- gsub("^Napali$|^Napaly$|^Nepal$|^Nepalese$|^Nipali$|^Nupali$",
                               "Napali", data$Language_56)
  data$Language_56 <- gsub("^Ojibway$",
                               "Ojibwe", data$Language_56)
  data$Language_56 <- gsub("^Ordu$",
                               "Urdu", data$Language_56)
  data$Language_56 <- gsub("^Orumo$",
                               "Oromo", data$Language_56)
  data$Language_56 <- gsub("^Oyiherero$",
                               "Otjiherero", data$Language_56)
  data$Language_56 <- gsub("^Pashato$|^Pashda$|^Pashtoo$|^Pashtu$|^Pastu$|^Pershdu$|^Pershta$|^Pshto$|^Pushto$|^Purshdu$|^Poshto$",
                               "Pashto", data$Language_56)
  data$Language_56 <- gsub("^Persain$|^Persan$|^Persian/Dari$|^Persian/French$|^Persion$|^Pirson$",
                               "Persian", data$Language_56)
  data$Language_56 <- gsub("^Philipino$|^Philippino$|^Phillipino$|^Philopeno$|^Philopino$",
                               "Filipino", data$Language_56)
  data$Language_56 <- gsub("^Portuguese/English$",
                               "Portuguese", data$Language_56)
  data$Language_56 <- gsub("^Polish/English$",
                               "Polish", data$Language_56)
  data$Language_56 <- gsub("^Segrenia$|^Serb-Croatia$|^Serb/Croatian$|^Serb/Croation$|^Serbia$|^Serbien$|^Serbin$|^Serian$",
                               "Serbian", data$Language_56)
  data$Language_56 <- gsub("^Botswana$",
                               "Setswana", data$Language_56)
  data$Language_56 <- gsub("^Shana$",
                               "Shona", data$Language_56)
  data$Language_56 <- gsub("^Sign$|^Asl$",
                               "Sign Language", data$Language_56)
  data$Language_56 <- gsub("^Sihalies$|^Singalese$|^Sinhalese$|^Sinhalasese$|^Sinhalese$|^Sinhaless$|^Sinhalish$|^Sinhanles$|^Srilaken$",
                               "Sinhala", data$Language_56)
  data$Language_56 <- gsub("^Ska$|^Skova$|^Slovaka$|^Slovakia$|^Slovakian$|^Slovac$",
                               "Slovak", data$Language_56)
  data$Language_56 <- gsub("^Somali/English$",
                               "Somali", data$Language_56)
  data$Language_56 <- gsub("^Tagalag$|^Tagalio$|^Tagallo$|^Tagalo$|^Tagalo/English$|^Taglog$|^Tagolog$|^Talalog$|^Talgalog$|^Tigalog$|^Tagalog$|^Talogh$|^Tegalo$|^Tegelo$|^Tigalo$|^Thigalo$|^Tangalo$",
                               "Filipino", data$Language_56)
  data$Language_56 <- gsub("^Tebrena$|^Tegrigna$|^Tegrina$|^Tegrino$|^Tgirinia$|^Tgrinia$|^Tigera$|^Tigeran$|^Tighna$|^Tighrina$|^Tigisty$|^Tigma$|^Tigrana$|^Tigrani$|^Tigre$|^Tigregna$",
                               "Tigrigna", data$Language_56)
  data$Language_56 <- gsub("^Tigregne$|^Tigrena$|^Tigrenia$|^Tigrgina$|^Tigrgna$|^Tigri$|^Tigria$|^Tigrian$|^Tigrigna$|^Tigrignha$|^Tigrina$",
                               "Tigrigna", data$Language_56)
  data$Language_56 <- gsub("^Tigrigna$|^Tigrinea$|^Tigrna$|^Tigrnea$|^Tirgcha$|^Tirgiary$|^Tirnayia$|^Triring$|^Tugrnia$|^Tigrigna$|^Tigring$|^Tigrinia$|^Tigrinya$",
                               "Tigrigna", data$Language_56)
  data$Language_56 <- gsub("^Trgir$|^Tirge$",
                               "Tigre", data$Language_56)
  data$Language_56 <- gsub("^Talugu$|^Telegu$|^Telugiu$",
                               "Telugu", data$Language_56)
  data$Language_56 <- gsub("^Teibetan$|^Tibetian$|^Tibetin$|^Tibian$",
                               "Tibetan", data$Language_56)
  data$Language_56 <- gsub("^Turkis$",
                               "Turkish", data$Language_56)
  data$Language_56 <- gsub("^Teranian$",
                               "Iranian", data$Language_56)
  data$Language_56 <- gsub("^Yoruda$",
                               "Yoruba", data$Language_56)
  data$Language_56 <- gsub("^Varsi$",
                               "Farsi", data$Language_56)
  data$Language_56 <- gsub("^Uzbec$|^Uzbeki$",
                               "Uzbek", data$Language_56)
  data$Language_56 <- gsub("^Shangu$",
                               "Sangu", data$Language_56)
  
  
  
  
  
  # numerics into numerics
  print("Convert numeric columns into numbers")
  # numeric (or need to be converted) from manual inspection
  dataNumeric <- c("Override_Et_Status_Cascadi_261",
                       "R_A_Override_Minutes_295",
                       "Pt_Weight_350")
  for (fac in dataNumeric) {
    print(fac)
    eval(parse(text=paste0("data$", fac, "<- as.numeric(as.character(data$", fac, "))")))
  }
  
  
  
  # ================================= AUTOMATICALLY SORT COLUMNS ================================= # 
  print("Examining Missingness")
  # check % of missingness in columns
  na_count <-sapply(data, function(y) round((sum(length(which(is.na(y))))/nrow(data))*100, 5))
  na_count <- data.frame(na_count); 
  na_count$colName <- rownames(na_count)
  rownames(na_count) <- NULL
  print(paste("There are", nrow(na_count %>% dplyr::filter(na_count == 100)), "columns with 100% missingness"))
  print(paste("There are", nrow(na_count %>% dplyr::filter(na_count == 0)), "columns with 0% missingness"))
  all.zeros <- as.character(unlist((na_count %>% dplyr::filter(na_count == 100))$colName))
  
  # check which columns are dates
  print("Matching Dates")
  date.pattern <-"^(19|(2(0|1)))([0-9]{6}|[0-9]{10}|[0-9]{12})((?!([0-9]+|\ |[A-Za-z])))"
  possible_dates <- sapply(data , function(y) any(grep(date.pattern,  y, perl=T)))
  possible_dates <- names(possible_dates[possible_dates==TRUE])
  print(length(possible_dates))
  
  # From manual examinatin, exclude these values
  possible_dates <- possible_dates[!possible_dates=='Pt_Accnt_5']
  possible_dates <- possible_dates[!possible_dates=='Hsc_7']
  possible_dates <- possible_dates[!possible_dates=='Ems_Id_32']
  possible_dates <- possible_dates[!possible_dates=='Health_Card_68']
  possible_dates <- possible_dates[!possible_dates=='Acct_Label_2_197']
  possible_dates <- possible_dates[!possible_dates=='Acct_Label_3_198']
  possible_dates <- possible_dates[!possible_dates=='Status_Admt_246']
  possible_dates <- possible_dates[!possible_dates=='Meds_Review_Prntd_278']
  possible_dates <- possible_dates[!possible_dates=='Pt_Lock_Boxed_D_T_282']
  possible_dates <- possible_dates[!possible_dates=='Glass_Broken_D_T_283']
  possible_dates <- possible_dates[!possible_dates=='Cpso_396']
  possible_dates <- possible_dates[!possible_dates=='Lic_549']
  possible_dates <- possible_dates[!possible_dates=='Post_It_Note_Arrival_612']
  possible_dates <- possible_dates[!possible_dates=='Vs_Acknowledge_618']
  possible_dates <- possible_dates[!possible_dates=='String_016_711']
  possible_dates <- possible_dates[!possible_dates=='String_021_712']
  possible_dates <- possible_dates[!possible_dates=='Emar_Trigger_745']
  
  print("saving dates")
  write.csv(possible_dates, "dates_colnames.csv")
  
  # BLOB
  print("Retrieving blob names")
  blob.pattern <- "\\(BLOB\\)"
  blob.pattern.matched <- sapply(data, function(y) any(grep(blob.pattern,  y)))
  sum(blob.pattern.matched=='TRUE')
  BLOBS <- names(blob.pattern.matched[blob.pattern.matched=='TRUE'])
  
  all.columns <- colnames(data)
  all.columns <- all.columns[!all.columns %in% c(all.zeros, possible_dates, BLOBS)]
  
  write.csv(all.columns, "other_colnames.csv")
  
  
  
  
  # ================================= PROCESS DATE COLUMNS ================================= #
  
  cat("\nProcessing dates\n")
  j <- 1
  length(possible_dates)
  for (date.col in possible_dates) {
    print(paste0(j, ": ", date.col))
    proc.date <- as.character(unlist(data[,c(date.col), with=FALSE]))
    proc.date <-as.character(sapply(proc.date, function(x) str_extract(x, date.pattern)))
    print(paste("Before Processing: ", unique(na.exclude(proc.date)[1])))
    if (nchar(unique(na.exclude(proc.date)[1]))==8) {
      proc.date <- as.POSIXct(strptime(proc.date, format="%Y%m%d"), tz="EST")
    } else if (nchar(unique(na.exclude(proc.date)[1]))==12) {
      proc.date <- as.POSIXct(strptime(proc.date, format="%Y%m%d%H%M"), tz="EST")
   } else {
      paste("Not Valid String Format")
    }
    print(paste("After Processing: ", unique(na.exclude(proc.date)[1])))
    eval(parse(text=paste0("data$", date.col, " <- proc.date")))
    
    j <- j + 1
  }

  
  # save processed wellSoft data to file for reading in future
  
  cat("\nWriting file\n") # remove BLOBS and empty columns when writing
  print(dim(data[,c(all.columns, possible_dates), with=FALSE]))
  fwrite(data[,c(all.columns, possible_dates), with=FALSE], paste0(file.path, "cleaned_wellSoft.csv"), dateTimeAs="write.csv")
  
  cat("\nDone\n")
  
  return(data)
  
}
