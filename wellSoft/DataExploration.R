#source('Shiny_functions.R')
library(ggplot2)
library(DT)
library(shiny)
library(dplyr)

max.columns <- 12 # maximum number of columns to display on univariate factor outputs

# Define UI for app that draws a histogram ----
ui <- fluidPage(
  
  # App title ----
  titlePanel("Emergency Department Data Exploration"),
  
  fluidRow(
    
    tabsetPanel(type="tabs",
                
                tabPanel("Variables",
                     column(4, 
                            div(selectizeInput('univariateColumns', label="",
                               choices = colnames(reg_codes)[(!colnames(reg_codes) %in% c("RegistrationNumber", "MedicalRecordNumber", "PrimaryMedicalRecordNumber", "IsMale",
                                                                                      "StartOfVisit", "EndOfVisit", "ReasonForVisit"))],
                               multiple=FALSE,
                               options=list(placeholder="Choose which columns to display"))),
                            
                            div(style = 'overflow-x: scroll; overflow-y: scroll', 
                                uiOutput('univariateOutputs')),
                      style="overflow-x: scroll; overflow-y: scroll; height:700px"), # end of side bar of Univariate Analysis
                     
                     column(8, 
                            plotOutput("univariatePlot",
                                 click = "plot_click",
                                 dblclick = "plot_dblclick",
                                 hover = "plot_hover",
                                 brush = "plot_brush"
                            ))
                    ), # end univariate analysis
                
                tabPanel("Number of Visits",
                         plotOutput("histoNumOfVisits")),
                
                tabPanel("ED Returns", column(4,
                             
                             sliderInput(inputId = 'histo_date_range',
                                          label = 'Number of Returns to ED: ',
                                          min = min(num.readmissions$num.returns), 
                                          max = max(num.readmissions$num.returns),
                                         value = c(min(num.readmissions$num.returns),
                                                   max(num.readmissions$num.returns))),
                             
                             sliderInput(inputId = "histo_diff_in_visits",
                                         label = "Minimum Length of Time between Visits (in Days):",
                                         min = 0, #round(min(time.lapse$DifferenceInDays))
                                         max = 300, #ceiling(max(time.lapse$DifferenceInDays))
                                         value = 300),

                             sliderInput(inputId = "bins",
                                         label = "Number of bins:",
                                         min = 5,
                                         max = 200,
                                         value = 30)
                             
                        ), # end input slider column
                    
                        column(8, 
                               
                               # Output: Histogram of Number of Visits to the ED 
                               plotOutput(outputId = "HistogramNumVisits", height=500,
                                          click="HistogramNumVisits_click",
                                          dblclick="HistogramNumVisits_dblclick",
                                          brush=brushOpts(
                                            id="HistogramNumVisits_brush", 
                                            resetOnNew = TRUE
                                          )
                               ),
                               
                               # Output: Summary statistics from Histogram 
                               verbatimTextOutput("HistogramSummaryStats")
                               
                        ) # end main histo box
                  ), # end ED returns panel
                
                tabPanel("Issues in data",
                         column(4, 
                            div(selectizeInput('issuesNames', label="",
                               choices = c("Length of Stay", "NA"),
                               multiple=FALSE)),
                            div(style = 'overflow-x: scroll; overflow-y: scroll', 
                                uiOutput('issuesOutputs')),
                            style="overflow-x: scroll; overflow-y: scroll; height:700px"
                            ), # end of side bar of Univariate Analysis
                         column(8, 
                            plotOutput("issuesPlot",
                                       click = "plot_click",
                                       dblclick = "plot_dblclick",
                                       hover = "plot_hover",
                                       brush = "plot_brush"
                                ))
                ), # end issues in Data tab
                
                #tabPanel("Map"), # end of length between visits tab
                
                
                tabPanel("Raw Data",
                         column(4, 
                                div(selectizeInput('columns.selected', label="",
                                                   choices = colnames(reg_codes)[(!colnames(reg_codes) %in% c("RegistrationNumber", "PrimaryMedicalRecordNumber"))],
                                                   multiple=TRUE,
                                                   options=list(placeholder="Choose which columns to display"))), 
                                div(numericInput("num.rows", "Number of Patients: ", 10, min=1, max=N.visits)),
                                div(radioButtons("type.of.sample", "Order of Patients:", 
                                                 c("Random Sample" = "rand.samp",
                                                   "Most Visits to ER" = "max.ed.visits",
                                                   "Least Visits to ER" = "min.ed.visits"))) # end radio buttons
                                ), # end side panel 
                         column(8, 
                                div(DT::dataTableOutput("RawDataTable")),
                                style="overflow-x: scroll; overflow-y: scroll; height:700px") 
                ) # end Raw Data TabPanel
      ) # end tabset panel
  ) # end fluidrow
) # end ui








server <- function(input, output) {
  
  # --------------- Univariate plots  ---------------- # 
  
  getDateRange <- reactive({input$univariateDateRange})
  getFactorColumns <- reactive({input$univariateFactorSelectize})
  getNumericRange <- reactive({input$univariateNumericSlider})
  
  output$univariatePlot <- renderPlot({
      uni.var <- input$univariateColumns
      date.range <- getDateRange()
      rel.factor.columns <- getFactorColumns()
      rel.numeric.range <- getNumericRange()
  
    if (class(reg_codes[, c(uni.var)])=="factor") {
      eval(parse(text=paste0("rel.reg_codes <- reg_codes %>% dplyr::filter(", uni.var, "%in% rel.factor.columns)")))

      eval(parse(text=paste0("uni.p <- ggplot(data=rel.reg_codes, aes(", uni.var, ")) + geom_bar() + theme_bw() + 
                             theme(axis.text.x=element_text(angle=45, hjust=1))")))


    } else if (class(reg_codes[, c(uni.var)]) == "numeric") {
        eval(parse(text=paste0("rel.reg_codes <- reg_codes %>% dplyr::filter(rel.numeric.range[1] <= ",
                    uni.var,  "& ", uni.var, "<= rel.numeric.range[2])")))
        eval(parse(text=paste0("uni.p <- ggplot(data=rel.reg_codes, aes(", uni.var, ")) + geom_histogram() + theme_bw()")))
    } else if (class(reg_codes[, c(uni.var)]) == "Date") {
        eval(parse(text=paste0("dat <- data.frame(reg_codes %>%
                     dplyr::group_by(", uni.var, ") %>%
                     dplyr::mutate(num.visits = n()))")))
       # if (!is.na(date.range))  { # date range has been set --> restrict output to date ranges
        eval(parse(text=paste0("dat <- dat %>% dplyr::filter(date.range[1] <= ", 
                          uni.var, "& ", uni.var, " <= date.range[2])")))
    #}
        
        dat <- dat[, c(uni.var, "num.visits")]
        eval(parse(text=paste0("dat <- dat[!duplicated(dat$", 
                   uni.var, "),]")))
        eval(parse(text=paste0("uni.p <- ggplot(data=dat, aes(x=dat$",
                    uni.var, ", y=num.visits)) + geom_point() +
                     xlab('", uni.var,"') + ylab('Number of Visits') +
                    theme_bw()")))
    } else if (class(reg_codes[, c(uni.var)]) == "POSIXct" | class(reg_codes[, c(uni.var)]) == "POSIXt") {
      
    } else {
      uni.p <- textOutput("No Valid Plot")
    }
    
    uni.p
  })
  
  


  
  # Dynamically generate type of selector output
  
  
  get.univar.fac.table <- function(uni.var) {

    temp <- data.frame(data.frame(ID=reg_codes[,c(uni.var)]) %>% group_by(ID) %>% summarise(no_rows = length(ID)))
                       
    return(DT::renderDataTable(DT::datatable(temp,
                          rownames = NULL, colnames=NULL, selection='none',
                          options = list(dom = 't', pageLength=1000, scrollX=TRUE, scrollY=TRUE))))
    
  }

  getUniVar <- reactive({input$univariateColumns})

  output$univariateOutputs <- renderUI({
    uni.var <- getUniVar() 
    uni.column.vec <- reg_codes[, c(uni.var)]
    
    if (class(uni.column.vec)=="factor") {
        uni.choices = levels(uni.column.vec)
        if (length(levels(uni.column.vec)) > max.columns) {
          uni.choices = uni.choices[1:max.columns]
        } 
        
        output$univariateFactorTable <- get.univar.fac.table(uni.var) 
        
        tagList(selectizeInput("univariateFactorSelectize", "Columns to Display on Chart", 
                       choices=levels(uni.column.vec), selected = uni.choices, multiple=TRUE, 
                       options=list(maxOptions=50)),
                DT::dataTableOutput("univariateFactorTable")
                ) # end taglist
      
    } else if (class(uni.column.vec) == "numeric") {
        min.val <- min(na.exclude(uni.column.vec)); max.val <- max(na.exclude(uni.column.vec))
        sliderInput("univariateNumericSlider", paste0(uni.var, " Range:"), 
                    min=min.val, max=max.val, 
                    value=c(min.val, max.val))
    } else if (class(uni.column.vec) == "Date") {
      dateRangeInput('univariateDateRange',
                     label = 'Date range: ',
                     start = min(uni.column.vec), end = max(uni.column.vec))
    }
  }) 

  
  
  # --------------- Histogram of number of visits ----------------- #
  
  
  output$histoNumOfVisits <- renderPlot({
    
    num.visits <- 
    ggplot(data=rel.num.readmissions, aes(num.returns)) + geom_histogram(bins=num.bins) +
      xlab("Number of Admissions to the ER") + 
      ylab("Number of Patients") + 
      ggtitle("Histogram of Number of Visits to the ER Since 2002") + 
      theme_bw() 
    
  })
  
  
  # --------------- Histogram of number of Return Visists  ---------------- # 
  
  #ranges <- reactiveValues(x = NULL, y = NULL) # used for zoom into histogram
  
  getHistoInputBins <- reactive({input$bins})
  getHistoNumRange <- reactive({input$histo_date_range})
  getHistoDiffInVisits <- reactive({input$histo_diff_in_visits})
  
  output$HistogramNumVisits <- renderPlot({
    
    num.bins <- getHistoInputBins()
    num.range <- getHistoNumRange()
    diff.in.visits <- getHistoDiffInVisits()
    
    rel.num.readmissions <- num.readmissions %>% dplyr::filter(num.range[1] <= num.returns & num.returns <= num.range[2])
    
    selected.patients <- as.vector(unique((time.lapse %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>% dplyr::filter(any(DifferenceInDays < diff.in.visits)))$PrimaryMedicalRecordNumber))
    
    rel.num.readmissions <- rel.num.readmissions %>% dplyr::filter(PrimaryMedicalRecordNumber %in% selected.patients)

    ggplot(data=rel.num.readmissions, aes(num.returns)) + geom_histogram(bins=num.bins) +
      xlab("Number of Admissions to the ER") + 
      ylab("Number of Patients") + 
      ggtitle("Histogram of Number of Visits to the ER Since 2002") + 
      theme_bw() 

  })
  

  # Displays summary statistics of data currently in view in histogram
  
  output$HistogramSummaryStats <- renderText({
    date.range <- getHistoNumRange()
    diff.in.visits <- getHistoDiffInVisits()

    rel.num.readmissions <- num.readmissions %>% dplyr::filter(num.returns >= date.range[1] & num.returns <= date.range[2])

    selected.patients <- as.vector(unique((time.lapse %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>% dplyr::filter(any(DifferenceInDays < diff.in.visits)))$PrimaryMedicalRecordNumber))
    
    rel.num.readmissions <- rel.num.readmissions %>% dplyr::filter(PrimaryMedicalRecordNumber %in% selected.patients)
    
    txt1 <- paste0("Displaying Number of Admissions from ", round(date.range[1]), " to ", round(date.range[2]))
    txt3 <- paste0("\nTotal Number of Patients: ", nrow(rel.num.readmissions), 
            "\nMean: ", round(mean(rel.num.readmissions$num.returns), 3), 
           "\nMedian: ", round(median(rel.num.readmissions$num.returns), 3))
      
 #   } else {
#      txt1 <- paste0("Displaying all data")
#      txt3 <- paste0("\nTotal Number of Patients: ", N.visits,
#              "\nMean: ", round(mean.visits, 3),
#             "\nMedian: ", round(median.visits, 3))
      
#    }
    
    # When a click happens, display number of observataions at that x value 
    if (!is.null(input$HistogramNumVisits_click$x)) {
      num.patients <- nrow(num.readmissions[num.readmissions$num.returns == round(input$HistogramNumVisits_click$x),])
      txt2 <- paste0("\nNumber of Patients with ", round(input$HistogramNumVisits_click$x), 
                    " visits to the ER: ", num.patients)
    } else {
      txt2 <- "\n"
    }
    
    paste0(txt1, txt2, txt3)

  })
  
  # ---------------------- Visualize issues in Data ------------------------ #
  
  getIssue <- reactive({input$issuesNames})
  getIssuesLenRange <- reactive({input$issuesLenOfStaySlider})
  getIssuesCalcLenRange <- reactive({input$issuesCalcLenOfStaySlider})
  getMaxDiffInLength <- reactive({input$issuesMaxDiff})
  
  output$issuesPlot <- renderPlot({
    issue <- getIssue()
  
    
    if (issue %in% c("Length of Stay")) {

      stend <- reg_codes[,c("RegistrationNumber", "VisitStartDate", "StartOfVisit", "EndOfVisit", "LengthOfStayInMinutes")]
      stend$CalculatedLengthOfStay <- as.numeric(difftime(stend$EndOfVisit, stend$StartOfVisit, units='mins'))
      stend <- na.exclude(stend)
      stend$Difference <- stend$CalculatedLengthOfStay - stend$LengthOfStayInMinutes
      
      len.range <- getIssuesLenRange()
      calc.len.range <- getIssuesCalcLenRange()
      max.diff <- getMaxDiffInLength()
      
      stend <- stend %>% dplyr::filter(abs(Difference) < max.diff)
      stend <- stend %>% dplyr::filter(len.range[1] < LengthOfStayInMinutes & LengthOfStayInMinutes < len.range[2])
      stend <- stend %>% dplyr::filter(calc.len.range[1] < CalculatedLengthOfStay & CalculatedLengthOfStay < calc.len.range[2])
      
      
      issues.plot <- ggplot(data=stend, aes(x = LengthOfStayInMinutes, y=CalculatedLengthOfStay)) + 
                    geom_point(aes(colour = VisitStartDate)) + xlab("Length of Stay In Minutes") + 
                  ylab("Calculated Length of Stay") + theme_bw() + geom_abline(slope=1, intercept=0)

    }
    
    issues.plot

    
  })
  
  
  
  output$issuesOutputs <- renderUI({
    
    issue <- getIssue() 
    
    if (issue == "Length of Stay") {
      
      stend <- reg_codes[,c("RegistrationNumber", "VisitStartDate", "StartOfVisit", "EndOfVisit", "LengthOfStayInMinutes")]
      stend$CalculatedLengthOfStay <- as.numeric(difftime(stend$EndOfVisit, stend$StartOfVisit, units='mins'))
      stend <- na.exclude(stend)
      stend$Difference <- stend$CalculatedLengthOfStay - stend$LengthOfStayInMinutes
      
      len.min <- min(stend$LengthOfStayInMinutes); len.max <- max(stend$LengthOfStayInMinutes)
      calc.min <- min(stend$CalculatedLengthOfStay); calc.max <- max(stend$CalculatedLengthOfStay)
      min.diff <- min(stend$Difference); max.diff <- max(stend$Difference)

      tagList(sliderInput("issuesLenOfStaySlider", "Length of Stay Range:", 
                  min=len.min, max=len.max, 
                  value=c(len.min, len.max)),
              sliderInput("issuesCalcLenOfStaySlider", "Calculated Length of Stay Range:", 
                          min=calc.min, max=calc.max, 
                          value=c(calc.min, calc.max)),
              numericInput("issuesMaxDiff", "Maximum difference between times: ", 10, min=min.diff, max=max.diff)
      ) # end taglist
      
    }
  })
  
  
  
  
  
  
  


  
  # ---------------------- Display Raw Data ------------------------ #
  getRawDataTable <- function(num.rows, columns.selected, type.of.sample) {
    random.cols <- c("RegistrationNumber", "PrimaryMedicalRecordNumber")
    
    if (type.of.sample == "rand.samp") {
      if (is.na(num.rows)) {num.rows <- 10}
      unique.PMRN <- sample(unique.PMRN, num.rows)
    } else if (type.of.sample == "min.ed.visits") {
      unique.PMRN <- rev(unique.PMRN)
    }
    
    
    if (!is.na(num.rows) && !is.null(columns.selected)) { # rows selected and columns selected
      display <- data.frame((reg_codes %>% dplyr::filter(PrimaryMedicalRecordNumber %in% unique.PMRN[1:num.rows])) %>% dplyr::group_by(PrimaryMedicalRecordNumber))
      columns.selected <- c(random.cols, columns.selected)
      display <- display[, columns.selected]
    } else if (!is.null(columns.selected)) { # only columns selected
      display <- data.frame((reg_codes[,random.cols] %>% dplyr::filter(PrimaryMedicalRecordNumber %in% unique.PMRN[1:10])) %>% dplyr::group_by(PrimaryMedicalRecordNumber))
      columns.selected <- c(random.cols, columns.selected)
      display <- display[, columns.selected]
    } else if (!is.na(num.rows)) { # only number of patients displayed selected
      display <- data.frame((reg_codes %>% dplyr::filter(PrimaryMedicalRecordNumber %in% unique.PMRN[1:num.rows])) %>% dplyr::group_by(PrimaryMedicalRecordNumber))
      display <- display[, random.cols]
    } else {
      display <- data.frame((reg_codes %>% dplyr::filter(PrimaryMedicalRecordNumber %in% unique.PMRN[1:10])) %>% dplyr::group_by(PrimaryMedicalRecordNumber))
      display <- display[, random.cols]
    }
    

    return(display)
  }

  raw.data.table <- reactive({getRawDataTable(input$num.rows, input$columns.selected, input$type.of.sample)})
  output$RawDataTable <- DT::renderDataTable({raw.data.table()},
                   options = list(lengthChange = FALSE, paging=FALSE,
                   sDom  = '<"top">lrt<"bottom">ip'),
                   rownames=FALSE)            
  
  # ------------------------------------------------------------------- #
}



shinyApp(ui = ui, server = server)
