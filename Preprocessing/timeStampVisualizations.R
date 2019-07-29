# July 29th, 2019

# Visualizations of Time Stamps of EPIC data 
## !! NOTE: Disch.Date.Time is time of discharge from hospital, NOT from ED

library(dplyr)
library(ggplot2)
library(ggpubr)
library(data.table)


# ===================== Load and Preprocess =====================

path <- "./data/EPIC_DATA/"

EPIC <- fread(paste0(path, "EPIC.csv"))
EPIC$Arrived <- as.POSIXct(EPIC$Arrived, tz="EST", format="%d/%m/%y %H%M")
EPIC <- EPIC[order(EPIC$Arrived),]
EPIC <- EPIC[!is.na(EPIC$Arrived),]
EPIC$Disch.Date.Time <- as.POSIXct(EPIC$Disch.Date.Time, tz="EST", format="%d/%m/%Y %H%M")

# ===================== (EXTRA) Explore Dates and Times =====================


EPIC %>% filter(is.na(Disch.Date.Time)) %>% 
  select(CSN, MRN, Disch.Date.Time, Discharge.Admit.Time, ED.Completed.Length.of.Stay..Hours.)

times <- EPIC[,c("CSN", "MRN", "Arrived", "Disch.Date.Time", "Primary.Dx", "CC", "Dispo")]

LengthOfStayInMinutes <-as.numeric(as.character(EPIC$ED.Completed.Length.of.Stay..Minutes.))


times$CalcDischargeTime <- EPIC$Arrived + (LengthOfStayInMinutes*60)
times$LengthOfStayInMinutes <- LengthOfStayInMinutes
times$CalculatedLengthOfStayInMinutes <- as.numeric(difftime(times$Disch.Date.Time, times$Arrived, units = "mins"))



filtered.times <- times %>% filter(LengthOfStayInMinutes < (60*24) & CalculatedLengthOfStayInMinutes < (60*24))

nrow(filtered.times)/nrow(times)

times$Error <- (times$LengthOfStayInMinutes - times$CalculatedLengthOfStayInMinutes) / times$LengthOfStayInMinutes

# ===================== Point: LoS vs CLoS by Arrival Times =====================
ggplot(data=times, 
       aes(x=LengthOfStayInMinutes, 
           y=CalculatedLengthOfStayInMinutes, 
           colour=Arrived)) + 
  geom_point()

ggplot(data=filtered.times, 
       aes(x=LengthOfStayInMinutes, 
           y=CalculatedLengthOfStayInMinutes, 
           colour=Arrived)) + 
  geom_point()

head(times)

# ===================== Point: Calculated Discharge Time vs Recorded Discharge Time =====================

ggplot(data=times, aes(x=Disch.Date.Time, y=CalcDischargeTime, colour=Dispo)) + 
  geom_point() + 
  theme_bw()


# ===================== Histogram: LoS vs CLoS =====================
p1 <- ggplot(times, aes(x=LengthOfStayInMinutes/60)) + 
  geom_histogram(bins = 100) + 
  theme_bw() + 
  ggtitle("Length Of Stay Captured In EPIC") + 
  xlab("Length Of Stay (in hours)"); p1
p2 <- ggplot(times, aes(x=CalculatedLengthOfStayInMinutes/60)) + 
  geom_histogram(bins=100) + 
  theme_bw() + 
  xlab("Length Of Stay (in hours)") + 
  ggtitle("Calculated Length Of Stay Based On Arrived and Disch.Date.Time"); p2

comb.hists <- ggarrange(p1, p2, nrow=2); comb.hists


# ===================== Histogram: LoS by Disposition and Primary.Dx =====================

p3 <- ggplot(times %>% filter(LengthOfStayInMinutes > 24*60), aes(x=LengthOfStayInMinutes/60, fill=Dispo)) + 
  geom_histogram(bins=40) + 
  xlab("Length Of Stay (in hours)")+ 
  theme_bw() + 
  ggtitle("Length Of Stay Captured in EPIC for Patients With Greater Than 24h Stays by Disposition"); p3


p4 <- ggplot(times %>% filter(LengthOfStayInMinutes > 24*60), aes(x=LengthOfStayInMinutes/60, fill=Primary.Dx)) + 
  geom_histogram(bins=40) + 
  xlab("Length Of Stay (in hours)")+ 
  theme_bw() + 
  ggtitle("Length Of Stay Captured in EPIC for Patients With Greater Than 24h Stays by Primary.Dx"); p4

causes_for_LoS <- ggarrange(p3, p4, nrow=2); causes_for_LoS


# ===================== Histogram: CLoS by Disposition =====================

p5 <- ggplot(times %>% filter(CalculatedLengthOfStayInMinutes > 24*60), 
             aes(x=CalculatedLengthOfStayInMinutes/60, fill=Dispo)) + 
  geom_histogram(bins=100) + 
  xlab("Calculated Length Of Stay (in hours)")+ 
  theme_bw() + 
  ggtitle("Calculated Length Of Stay for Patients With Greater Than 24h Stays by Disposition"); p5

# ===================== LoS Q1 and CLoS Q4 =====================

strange <- times %>% filter(LengthOfStayInMinutes < as.numeric(quantile(LengthOfStayInMinutes, na.rm=T)[2]))
strange <- strange %>% filter(CalculatedLengthOfStayInMinutes > as.numeric(quantile(times$CalculatedLengthOfStayInMinutes, na.rm=T)[4]))
nrow(strange)


ggplot(strange, aes(x=LengthOfStayInMinutes/60, y=CalculatedLengthOfStayInMinutes/60, colour=Dispo)) + 
  geom_point(size=3) + 
  xlab("Length Of Stay (in hours)") + 
  ylab("Calculated LengthOfStay (in hours)") + 
  theme_bw() + 
  theme(legend.position="bottom") + 
  ggtitle("Patients in Bottom Quartile of Length Of Stay and Top Quartile of Calculated Length Of Stay by Disposition")

# ===================== Point:LoS vs CLoS by Dispo =====================
ggplot(data=times, 
       aes(x=LengthOfStayInMinutes, 
           y=CalculatedLengthOfStayInMinutes, 
           colour=Dispo)) + 
  geom_point() + 
  theme_bw()+ 
  ggtitle("Length Of Stay versus Calculated Length Of Stay by Dispo")



ggplot(data=filtered.times, 
       aes(x=LengthOfStayInMinutes, 
           y=CalculatedLengthOfStayInMinutes, 
           colour=Dispo)) + 
  geom_point() + 
  theme_bw() + 
  ggtitle("Length Of Stay versus Calculated Length Of Stay by Dispo")

