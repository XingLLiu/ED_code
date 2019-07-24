
rm(list=ls())

pckgs <- c('data.table','stringr','magrittr','forcats','cowplot')
for (pp in pckgs) { library(pp,character.only = T) }

dir <- '/home/erik/Documents/projects/ED'
setwd(dir)

# Load data
dat <- fread('EPIC.csv')

# Load the CC1
dat[, complaint := str_replace_all(str_squish(str_remove_all(tolower(CC.1),'\\?')),'\\s\\/\\s','/')]
dat[, Date := as.Date(str_split_fixed(Arrived,'\\s',2)[,1],format='%d/%m/%y')]
dat[, Month := format(Date,'%b')]

# Get the total count
df.complaint <- dat[,list(n=.N),by=complaint][order(-n)]
# And by the month
df.complaint.month <- dat[,list(n=.N),by=list(complaint,Month)][order(-n)]
# Get the percent
df.complaint[, tot := sum(n)]
df.complaint.month[,tot := sum(n),by=Month]
df.complaint.month[, Month := factor(Month,levels=month.abb)]

# --- repeat for diagnosis --- # 
df.diagnosis <- dat[,list(n=.N),by=Primary.Dx][order(-n)]
df.diagnosis.month <- dat[,list(n=.N),by=list(Primary.Dx,Month)][order(-n)]
df.diagnosis[, tot := sum(n)]
df.diagnosis.month[,tot := sum(n),by=Month]
df.diagnosis.month[, Month := factor(Month,levels=month.abb)]

dat$ED.Complaint[c(30,40)]
dat$CC[c(30,40)]
dat$CC.1[c(30,40)]

# --- Make plots --- #
ntop <- 20
gg.complaint <- 
ggplot(df.complaint[1:ntop],aes(y=fct_reorder(complaint,n),x=n / tot * 100)) + 
  geom_point() + background_grid(major='xy',minor='none') + 
  theme(axis.title.y = element_blank()) + 
  labs(x='Share (%)',title='% of ED-EPIC visits by complaint (top 20)',
       subtitle = str_c('Total # of visits: ',nrow(dat))) + 
  scale_x_continuous(limits=c(0,25),breaks=seq(0,25,5)) + 
  geom_text(aes(label=str_c(round(n/1000,1),'K')),nudge_x = 1.5)
save_plot(filename=file.path(dir,'output','complaints.png'),plot = gg.complaint,base_height = 8,base_width = 8)

gg.diagnosis <- 
ggplot(df.diagnosis[1:ntop],aes(y=fct_reorder(Primary.Dx,n),x=n / tot * 100)) + 
  geom_point() + background_grid(major='xy',minor='none') + 
  theme(axis.title.y = element_blank()) + 
  labs(x='Share (%)',title='% of ED-EPIC visits by Primary Diagnosis (top 20)',
       subtitle = str_c('Total # of visits: ',nrow(dat))) +
  # scale_x_continuous(limits=c(0,25),breaks=seq(0,25,5)) + 
  geom_text(aes(label=str_c(round(n/1000,1),'K')),nudge_x = 0.5)
save_plot(filename=file.path(dir,'output','diagnoses.png'),plot = gg.diagnosis,base_height = 8,base_width = 8)

# Repeat for the seasons
n.seas <- 7
t.comp <- df.complaint[1:n.seas]$complaint
t.diag <- df.diagnosis[1:n.seas]$Primary.Dx

gg.complaint.month <- 
ggplot(df.complaint.month[complaint %in% t.comp & !is.na(Month)],
       aes(y=n / tot * 100,x=Month,color=complaint,group=complaint)) + 
  geom_point() + geom_line() + 
  background_grid(major='xy',minor='none') + 
  labs(y='Share (%)',title='% of ED-EPIC visits by Compaint per month') +
  theme(axis.title.x = element_blank(),legend.position = 'bottom',
        legend.justification = 'center') +
  scale_color_discrete(name='')
save_plot(filename=file.path(dir,'output','complaints_month.png'),plot = gg.complaint.month,base_height = 8,base_width = 8)

gg.diagnosis.month <-
  ggplot(df.diagnosis.month[Primary.Dx %in% t.diag & !is.na(Month)],
         aes(y=n / tot * 100,x=Month,color=Primary.Dx,group=Primary.Dx)) + 
  geom_point() + geom_line() + 
  background_grid(major='xy',minor='none') + 
  labs(y='Share (%)',title='% of ED-EPIC visits by Diagnosis per month') +
  theme(axis.title.x = element_blank(),legend.position = 'bottom',
        legend.justification = 'center') +
  scale_color_discrete(name='')
save_plot(filename=file.path(dir,'output','diagnosis_month.png'),plot = gg.diagnosis.month,base_height = 8,base_width = 8)



# dat <- dat[,1:50] # remove notes
# dat[, Date := as.Date(str_split_fixed(Arrived,'\\s',2)[,1],format='%d/%m/%y')]
# # Clean up the complaints
# dat[, ED.Complaint := ifelse(str_length(ED.Complaint)==0,as.character(NA),ED.Complaint)]
# dat[, ED.Complaint := stringi::stri_enc_toutf8(str=ED.Complaint) ]
# # Subset: note CSN is unique and MRN repeats for the same patient
# dat <- dat[,c('CSN','MRN','Date','ED.Complaint')]
# dat[, Month := format(Date,'%b')]
# dat[, `:=` (Complaint = ED.Complaint, ED.Complaint = NULL)]
# dat[, vals := str_split(Complaint,'\\,')]
# complaints <- unlist(with(dat,str_split(Complaint,'\\,')))
# # Trim
# complaints <- str_squish(complaints)
# # Remove NAs
# complaints <- complaints[!is.na(complaints)]
# # put everything ot lower
# complaints <- tolower(complaints)
# # remove non letters
# complaints[str_detect(complaints,'[^a-z\\s\\-]')] %>% head(100)
# complaints[str_count(complaints,'\\s')>4] %>% head(200)
# complaints[str_detect(complaints,'\\sand\\s')] %>% head(100)
