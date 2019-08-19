library(ggplot2)
library(DT)
library(shiny)
library(dplyr)
library(reshape2)
library(shinyWidgets)

## NEED TO CHECK IF WORKS!!

EPIC <- fread("EPIC.csv")

ui <- fluidPage(
  
  # App title ----
  titlePanel("EPIC Data"),
  
  fluidRow(
    
    tabsetPanel(type="tabs",
                tabPanel("Outline"),
                tabPanel("Raw Variables",
                         column(4, 
                                div(selectizeInput('univariateColumns', label="",
                                                   choices = colnames(EPIC)[(!colnames(EPIC) %in% c("X", "X.1", "Encounter.Number", "CSN", "Registration.Number",
                                                                                                              "MRN", "Address", "Colour", "Arrived", "Roomed"))],
                                                   multiple=FALSE,
                                                   options=list(placeholder="Choose which columns to display"))),
                                
                                div(style = 'overflow-x: scroll; overflow-y: scroll', 
                                    uiOutput('univariateOutputs')),
                                style="overflow-x: scroll; overflow-y: scroll; height:700px"), # end of side bar of Univariate Analysis
                         
                         column(8, 
                                div(plotOutput("univariatePlot",
                                           click = "plot_click",
                                           dblclick = "plot_dblclick",
                                           hover = "plot_hover",
                                           brush = "plot_brush"),
                                    style=" height:700px"),
                                div(materialSwitch(inputId = "univariateColourCoded", label = "Segment by Sepsis prediction?", status = "danger"),
                                    materialSwitch(inputId = "univariateNoSepsis", label = "Remove patients without Sepsis?", status = "danger")
                                    )
                                )
                      ) # close raw variables tab
                ) # close tabset panel
    ) # close fluidrow
) # close ui
  
 



server <- function(input, output) {
  
  
  
  max.columns <- 10
  
  getDateRange <- reactive({input$univariateDateRange})
  getFactorColumns <- reactive({input$univariateFactorSelectize})
  getNumericRange <- reactive({input$univariateNumericSlider})
  getColourCoded <- reactive({input$univariateColourCoded})
  getWithNoSepsis <- reactive({input$univariateNoSepsis})
  
  
  output$univariatePlot <- renderPlot({
    uni.var <- input$univariateColumns
    date.range <- getDateRange()
    rel.factor.columns <- getFactorColumns()
    rel.numeric.range <- getNumericRange()
    colour.coded <- getColourCoded()
    with.no.sepsis <- getWithNoSepsis()

    
    if (class(EPIC[, c(uni.var)])=="factor") {


      if (colour.coded) {
        
        if (with.no.sepsis) {
          eval(parse(text=paste0("rel.EPIC <- EPIC %>% dplyr::filter(True_Sepsis ==1)")))
          eval(parse(text=paste0("rel.EPIC <- rel.EPIC %>% dplyr::filter(", uni.var, "%in% rel.factor.columns)")))
        } else {
          eval(parse(text=paste0("rel.EPIC <- EPIC %>% dplyr::filter(", uni.var, "%in% rel.factor.columns)")))
        }
          eval(parse(text=paste0("rel.EPIC$", uni.var, " <- factor(rel.EPIC$", uni.var, ", levels=unique(rel.EPIC$", uni.var,"))")))
          
          eval(parse(text=paste0("rel.EPIC <- melt(table(rel.EPIC[,c('", uni.var,"', 'Colour')]))")))
          eval(parse(text=paste0("uni.p <- ggplot(data=rel.EPIC, aes(x=reorder(",uni.var,", -value), y = value, fill=Colour)) + geom_bar(stat='identity') + theme_bw() + 
                             theme(axis.text.x=element_text(angle=90, hjust=1)) + xlab('", uni.var,"') +
                                 theme(legend.position='bottom')")))
          
        
      } else {
        if (with.no.sepsis) {
          eval(parse(text=paste0("rel.EPIC <- EPIC %>% dplyr::filter(True_Sepsis ==1)")))
          eval(parse(text=paste0("rel.EPIC <- rel.EPIC %>% dplyr::filter(", uni.var, "%in% rel.factor.columns)")))
        } else {
          eval(parse(text=paste0("rel.EPIC <- EPIC %>% dplyr::filter(", uni.var, "%in% rel.factor.columns)")))
        }
        eval(parse(text=paste0("rel.EPIC$", uni.var, " <- factor(rel.EPIC$", uni.var, ", levels=unique(rel.EPIC$", uni.var,"))")))
        eval(parse(text=paste0("rel.EPIC <- melt(sort(table(as.character(rel.EPIC[,c('", uni.var,"')])), decreasing=TRUE))")))
        eval(parse(text=paste0("uni.p <- ggplot(data=rel.EPIC, aes(x=reorder(Var1, -value), y = value)) + geom_bar(stat='identity') + theme_bw() + 
                             theme(axis.text.x=element_text(angle=90, hjust=1)) + xlab('", uni.var,"') +
                               theme(legend.position='bottom')")))
              }
    } else if (class(EPIC[, c(uni.var)]) == "numeric") {
      eval(parse(text=paste0("rel.EPIC <- EPIC %>% dplyr::filter(rel.numeric.range[1] <= ",
                             uni.var,  "& ", uni.var, "<= rel.numeric.range[2])")))
      eval(parse(text=paste0("uni.p <- ggplot(data=rel.EPIC, aes(", uni.var, ")) + geom_histogram() + theme_bw()")))
    } else if (class(EPIC[, c(uni.var)]) == "Date") {
      eval(parse(text=paste0("dat <- data.frame(EPIC %>%
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
    } else if (class(EPIC[, c(uni.var)]) == "POSIXct" | class(EPIC[, c(uni.var)]) == "POSIXt") {
      
    } else {
      uni.p <- textOutput("No Valid Plot")
    }
    
    uni.p
  })
  

  
  
  # Dynamically generate type of selector output
  
  
  get.univar.fac.table <- function(uni.var, noSepsis) {
    if (noSepsis) {
      temp <- data.frame(data.frame(ID=EPIC[EPIC$True_Sepsis==1,c(uni.var)]) %>% group_by(ID) %>% summarise(no_rows = length(ID)))
    }
    temp <- data.frame(data.frame(ID=EPIC[,c(uni.var)]) %>% group_by(ID) %>% summarise(no_rows = length(ID)))
    temp <- temp[order(temp$no_rows, decreasing=TRUE),]
    
    return(DT::renderDataTable(DT::datatable(temp,
                                             rownames = NULL, colnames=NULL, selection='none',
                                             options = list(dom = 't', pageLength=1000, scrollX=TRUE, scrollY=TRUE))))
    
  }
  
  getUniVar <- reactive({input$univariateColumns})
 #getWithNoSepsis <- reactive({input$univariateNoSepsis})
  

  
  output$univariateOutputs <- renderUI({
    uni.var <- getUniVar()
    uni.column.vec <- EPIC[, c(uni.var)]
    noSepsis <- getWithNoSepsis()
    
    if (class(uni.column.vec)=="factor") {

      if (noSepsis) {
        uni.column.vec <- EPIC[EPIC$True_Sepsis==1, c(uni.var)]
      } 
        uni.choices = names(sort(table(uni.column.vec), decreasing=TRUE))
    
      
      
      if (length(uni.choices) > max.columns) {
        uni.choices = uni.choices[1:max.columns]
      } 
      
      output$univariateFactorTable <- get.univar.fac.table(uni.var, noSepsis) 
      
      tagList(selectizeInput("univariateFactorSelectize", "Columns to Display on Chart", 
                             choices=names(sort(table(uni.column.vec), decreasing=TRUE)), 
                             selected = uni.choices, 
                             multiple=TRUE, 
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
  
}

shinyApp(ui, server)