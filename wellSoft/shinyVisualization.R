library(shiny)
library(shinydashboard)

description.body <- dashboardBody(
            h1("Visualization of EPIC Data"),
            h5("This dashboard lets you explore EPIC data")
)

univariate.body <- dashboardBody(
  fluidRow(h1("Univariate Analysis")),
  fluidRow(
    box(title="Graph", width=NULL)
  ),
  fluidRow(
    column(width=4,
           box(title="Variable", solidHeader=TRUE, status='success', width=NULL,
               selectizeInput('univariateColumns', label="",
                              choices = colnames(EPIC)[(!colnames(EPIC) %in% c("X", "X.1", "Encounter.Number", "CSN", "Registration.Number",
                                                                               "MRN", "Address", "Colour", "Arrived", "Roomed", "CTAS"))],
                              multiple=FALSE)),
           box(title="Levels", solidHeader=TRUE, status='success', width=NULL,
               uiOutput('univariateOutputs'))),
    column(width=8,
        box(title="Variable Setting", solidHeader = TRUE, status = "warning", width=NULL),
        box(title="Options", solidHeader = TRUE, status = "primary", width=NULL)
      
    )
  )
  )


ui <- dashboardPage(
  dashboardHeader(title="EPIC Data"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Description of Data", tabName='description', icon=icon('dashboard')),
      menuItem("Univariate Analysis", tabName='univariate', icon=icon('th'))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName='description',
              description.body
      ),
      
      # Univariate
      tabItem(tabName='univariate',
              univariate.body)

    )
  )
)


server <- function(input, output) { }

shinyApp(ui, server)