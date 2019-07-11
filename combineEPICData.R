processClinNotes <- function(clinical.notes, note.types) {
  # For each note type, for each unique visit, combine all the notes 
  # For each patient visit:
  # FORMAT:
  #         - Patient.Name
  #         - CSN
  #         - MRN
  #         - Note.Data for each type of note: 
  #                 - Note types: "Note.Data_ED.Notes", "Note.Data_ED.Procedure.Note", "Note.Data_ED.Provider.Notes", "Note.Data_ED.Triage.Notes" 
  #                 - Format: <<STARTNOTE note.time, note.author.service, note.author <NOTETEXT ... NOTETEXT>ENDNOTE>>
  
  final.notes <- data.frame()
  i <- 1
  for (note.type in note.types) {
    print(note.type)
    note.data <- clinical.notes %>% filter(Note.Type == note.type)
    
    max.num.notes.per.pat <- max(table(note.data$CSN))
    
    
    new.note.data <- data.frame(matrix(nrow=length(unique(note.data$CSN)), ncol=4))
    colnames(new.note.data) <- c("Patient.Name", "CSN", "MRN", paste0("Note.Data_", make.names(note.type)))
    
    print(paste("Total num rows:", nrow(note.data)))
    print(paste("Unique visits:", length(unique(note.data$CSN))))
    print(paste("Max num per encounter:", max.num.notes.per.pat))
    print(head(sort(table(note.data$CSN), decreasing = T)))
    
    j <- 1
    for (csn.num in unique(note.data$CSN)) {
      pat.data <- note.data[note.data$CSN == csn.num,]
      if (nrow(pat.data) > 10) {
        print(paste("Number rows of Patient Data:", nrow(pat.data)))
      }
      pat.name <- ifelse(length(pat.data$Patient.Name) > 1, pat.data$Patient.Name[1], pat.data$Patient.Name)
      pat.MRN <- ifelse(length(pat.data$MRN) > 1, pat.data$MRN[1], pat.data$MRN)
      
      pat.plain.text <- ""
      for (row.num in 1:nrow(pat.data)) {
        file.time <- pat.data$File.Time[row.num]
        author.service <- pat.data$Author.s.Service[row.num]
        note.author <- pat.data$Note.Author[row.num]
        pat.plain.text <- paste0(pat.plain.text, 
                                 "<<STARTNOTE ", 
                                 file.time, ",",
                                 author.service,",",
                                 note.author,
                                 "<NOTETEXT ",
                                 pat.data$Note.Plain.Text..HNO.,
                                 " NOTETEXT>ENDNOTE>>")
      }
      
      new.note.data[j,] <- c(pat.name, as.numeric(as.character(csn.num)), as.numeric(as.character(pat.MRN)), pat.plain.text)
      j <- j + 1
      
    }
    print(colnames(new.note.data)); print(dim(new.note.data))
    if (i == 1) {
      final.notes <- data.frame(new.note.data)
      final.notes$CSN <- as.numeric(as.character(final.notes$CSN))
      final.notes$MRN <- as.numeric(as.character(final.notes$MRN))
    } else {
      print("merging data")
      new.note.data$CSN <- as.numeric(as.character(new.note.data$CSN))
      new.note.data$MRN <- as.numeric(as.character(new.note.data$MRN))
      
      final.notes <- data.frame(merge(x=final.notes,
                                      y=new.note.data,
                                      by.x=c("Patient.Name", "CSN", "MRN"),
                                      by.y=c("Patient.Name", "CSN", "MRN"),
                                      all=TRUE))
    }
    i <- i + 1
    print("===========================================")
  }
  return (final.notes)
  
}



path <- "./data/EPIC_DATA/"



# ============ LOAD DATA SETS ================= #


# 1. Load EPIC Data from Aug 18 - Feb 19
EPIC <- fread(paste0(path, "ED_DATA_EPIC_AUG18_TO_FEB19.csv"))
dim(EPIC)
EPIC$X <- NULL
EPIC$V1 <- NULL

# 2. Load new EPIC data from Feb 19 - June 2019(Some overlap on Feb 19th!)
data.filenames <- unlist(lapply(list.files(paste0(path, "EPIC_DATA_CSV/")), 
                                function(x) paste0(path, "EPIC_DATA_CSV/", x)))

NEW.EPIC <- data.frame(rbindlist(lapply(data.filenames, function(x) fread(x))))
dim(NEW.EPIC)
print(all(colnames(EPIC) == colnames(NEW.EPIC)))

# 3. Load clinical notes
notes.filenames <- unlist(lapply(list.files(paste0(path, "EPIC_NOTES_CSV/")), 
                                 function(x) paste0(path, "EPIC_NOTES_CSV/", x)))

clin.notes <- data.frame(rbindlist(lapply(notes.filenames, function(x) fread(x))))
dim(clin.notes)
colnames(clin.notes)

# 4. Load Geospatial Variables
dist_to_SK <- read.dbf(paste0(path, "Geospatial/EPIC_Distance_To_SickKids.dbf"))

dist_to_walkin <- read.dbf(paste0(path, "Geospatial/EPIC_Distance_To_Pediatric_Walkins.dbf"))

dist_to_hosp <- read.dbf(paste0(path, "Geospatial/EPIC_Distance_To_Hospitals.dbf"))

# ============ 1. & 2. PROCESS NEW EPIC DATA ================= #

# Add remainder of EPIC data to previously processed EPIC data
dup.visits <- intersect(EPIC$Registration.Number, NEW.EPIC$Registration.Number)

# the newer entries of the duplicates are used
EPIC <- rbind(EPIC[!EPIC$Registration.Number %in% dup.visits,], NEW.EPIC)
dim(EPIC)

# ============ 3. PROCESS EPIC CLINICAL NOTES ================= #

clin.notes$MRN <- as.numeric(as.character(clin.notes$MRN))

#remove duplicate clin notes
clin.notes <- clin.notes[!duplicated(clin.notes),] # remove duplicate rows

not.in.EPIC.data <- setdiff(unique(clin.notes[,c("CSN", "MRN")]), unique(EPIC[,c("CSN", "MRN")]))
clin.notes <- clin.notes %>% filter(!CSN %in% not.in.EPIC.data$CSN) # remove clin notes that do not appear in EPIC data
dim(clin.notes)

note.types <- c("ED Notes", "ED Procedure Note", "ED Provider Notes", "ED Triage Notes")

final.notes <- processClinNotes(clin.notes, note.types)

# Process EPIC Data more by removing duplicate CSN numbers 
# (second version has diagnoses, first duplicated CSN entry does not have diagnosis)
dup.CSN <- EPIC$CSN[duplicated(EPIC$CSN)];
non.dup.EPIC <- EPIC[!EPIC$CSN %in% dup.CSN,]
correct.dups <- EPIC %>% filter(CSN %in% dup.CSN) %>% arrange(CSN)
correct.dups <- correct.dups[duplicated(correct.dups$CSN),]
EPIC <- rbind(non.dup.EPIC, correct.dups)



# ============ 4. PROCESS GEOSPATIAL DATA ================= #

dist_to_SK <- dist_to_SK[,c("Address", "CSN", "MRN", "Distance_1")]
colnames(dist_to_SK) <- c("Address", "CSN", "MRN", "Distance_To_SickKids")
head(dist_to_SK)

dist_to_walkin <- dist_to_walkin[,c("Address", "CSN", "MRN", "Name_of_Cl", "Distance_2")]
colnames(dist_to_walkin) <- c("Address", "CSN", "MRN", "Name_Of_Walkin", "Distance_To_Walkin")
head(dist_to_walkin)

dist_to_hosp <- dist_to_hosp[,c("Address", "CSN", "MRN", "Name", "Distance_1")]
colnames(dist_to_hosp) <- c("Address", "CSN", "MRN", "Name_Of_Hospital", "Distance_To_Hospital")
head(dist_to_hosp)


geospatial <- merge(x=dist_to_SK,
                    y=dist_to_walkin,
                    by.x = c("Address", "CSN", "MRN"),
                    by.y=c("Address", "CSN", "MRN"))

geospatial <- merge(x=geospatial,
                    y=dist_to_hosp,
                    by.x = c("Address", "CSN", "MRN"),
                    by.y=c("Address", "CSN", "MRN"))


# ============ 5. Merge all data into EPIC ================= #

# Finally, merge EPIC data with processed clinical notes
print(nrow(EPIC))
EPIC <- merge(x=EPIC,
              y=final.notes[,c("CSN", "MRN", "Note.Data_ED.Notes", "Note.Data_ED.Procedure.Note", "Note.Data_ED.Provider.Notes", "Note.Data_ED.Triage.Notes")],
              by.x=c("CSN", "MRN"),
              by.y=c("CSN", "MRN"),
              all.x=TRUE)
print(nrow(EPIC))

EPIC <- merge(x=EPIC,
              y=geospatial, 
              by.x=c("CSN", "MRN", "Address"),
              by.y=c("CSN", "MRN", "Address"), 
              all.x=TRUE)

print(colnames(EPIC))

fwrite(x = EPIC, file = paste0(path, "EPIC.csv"))


