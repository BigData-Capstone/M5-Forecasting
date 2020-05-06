## Capstone Project
# M5 - Forecasting
# Link zur Challenge: https://www.kaggle.com/c/m5-forecasting-accuracy

# to save the current RData to the .RData
save.image("~/R-Code/M5-Forecasting/.RData")



#####################################################################
## OSEMN Pipeline in Data Science
#####################################################################
# nach https://medium.com/breathe-publication/life-of-data-data-science-is-osemn-f453e1febc10
#
# O - Obtaining our data
# S - Scrubbing / Cleaning our data
# E - Exploring / Visualizing our data (allow us to find patterns and trends)
# M - Modeling our data will give us our predictive pwer as a wizard
# N - Interpreting our data
#
#####################################################################


###@Flo: Stark angelehnt an https://www.kaggle.com/sebastiancarino/m5-forecasting-accuracy-r
#####################################################################
## O - Obtaining the data
#####################################################################
## Load the libraries
library(reshape2) ## for data cleaning
library(data.table) ## for quick and RAM efficient loading - install.packages("data.table")
library(tidyr) ## for data cleaning
library(tidyverse) ## for data cleaning and piping - install.packages("tidyverse")
library(lubridate) ## for date class transformation
library(splitstackshape) ## for stratified sampling - install.packages("splitstackshape")
library(ggplot2) ## for data visualizations and exploration

##Loading the files
sell_prices = fread("input/sell_prices.csv",
                    sep = ",",
                    na.strings = c("NA", ""),
                    header = TRUE, stringsAsFactors = TRUE)

calendar = fread("input/calendar.csv",
                    sep = ",",
                    na.strings = c("NA", ""),
                    header = TRUE, stringsAsFactors = TRUE)

train = fread("input/sales_train_validation.csv",
                    sep = ",",
                    na.strings = c("NA", ""),
                    header = TRUE, stringsAsFactors = TRUE)

#####################################################################

#####################################################################
## S - Scrubbing
#####################################################################



#####################################################################























#####################################################################
## Render & Forward defined results to RMarkdown File
#####################################################################
library(rmarkdown)
library(knitr)
render("main.Rmd",
      output_format = "html_document",
      output_file = "Capstone.html")
