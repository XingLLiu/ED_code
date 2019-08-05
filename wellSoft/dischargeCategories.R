# Categories of discharge disposition in registration codes data, as sorted by Olivia

to.remove <- c("NULL",
               "NO ENTRY-QUICK RELEASE",  
               "DIRECT ADMISSION",
               "ALREADY ADMITTED",
               "LEFT AGAINST MEDICAL ADVICE")

left.labels <- c("LEFT BEFORE SEEN BY DR",
                 "LEFT BEFORE SEEN BY DOCTOR",
                 "LEFT BEFORE FULL TRIAGE",
                 "LEFT BEFORE TRIAGE")

admitted.labels <- c("ADMIT TO HOLD - SENT HOME", # sent home after intervention
                     "SENT HOME-EMERG. ADMIT",
                     "SENT HOME VIA O.R.",
                     "ADMITTED TO HSC", # Admitted to Sick Kids
                     "ADMIT VIA O.R.", 
                     "ADMITTED TO CCU", 
                     "ADMITTED VIA O.R.",
                     "ADMITTED TO CCU OR THE O.R.",
                     "TRNSFR TO D/S",
                     "TRANSFER TO DAY SURGERY",
                     "TRANSFERRED-EMERG. ADMIT",
                     "TRANSFER TO ANOTHER INSTITUTION", # sent to another institution
                     "OTHER INSTITUTION",
                     "Transfer to another facility",
                     "ADMIT TO HOLD - TRANSFERRED",
                     "DEATH AFTER ARRIVAL", # Death
                     "DEAD ON ARRIVAL",
                     "EXPIRED",
                     "ADMIT TO HOLD - EXP.")

home.labels <- c("SENT HOME",
                 "HOME",
                 "SENT TO HSC CLINIC",
                 "DISCHARGED WITH HOMECARE",
                 "TRANSFER TO CLINIC",
                 "PLACE OTHER THAN HOME")