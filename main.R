## Capstone Project
# M5 - Forecasting
# Link zur Challenge: https://www.kaggle.com/c/m5-forecasting-accuracy

# to save the current RData to the .RData
save.image("~/R-Code/M5-Forecasting/.RData")

kjh

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



#####################################################################
## O - Obtaining the data - copy of https://www.kaggle.com/sebastiancarino/m5-forecasting-accuracy-r/notebook
#####################################################################
## Load the libraries
library(reshape2) ## for data cleaning
library(data.table) ## for quick and RAM efficient loading - install.packages("data.table")
library(tidyr) ## for data cleaning
library(tidyverse) ## for data cleaning and piping - install.packages("tidyverse")
library(lubridate) ## for date class transformation
library(splitstackshape) ## for stratified sampling - install.packages("splitstackshape")
library(ggplot2) ## for data visualizations and exploration

## Loading the files - It is done once and then availbe in our .RData (updated via Git - see code line 6)
sell_prices = fread("input/sell_prices.csv",sep = ",", na.strings = c("NA", ""), header = TRUE, stringsAsFactors = TRUE)
calendar = fread("input/calendar.csv", sep = ",", na.strings = c("NA", ""),  header = TRUE, stringsAsFactors = TRUE)
train = fread("input/sales_train_validation.csv", sep = ",", na.strings = c("NA", ""), header = TRUE, stringsAsFactors = TRUE)

#####################################################################

#####################################################################
## S - Scrubbing / Cleaning the data - angelehnt an  https://www.kaggle.com/sebastiancarino/m5-forecasting-accuracy-r/notebook
#####################################################################
## Checking out the dimensions of each data
data.frame(calendar = dim(calendar),
           sell_prices = dim(sell_prices),
           train = dim(train),
           row.names = c("Rows", "Columns"))


## Getting a glimpse of the datasets
head(calendar, n = 3)
head(sell_prices, n = 3)
head(train, n = 3)

## Transforming the train set into the right structure by transferring the d columns into rows
train = train %>% 
       melt(id.vars = c("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"),
       variable.name = "d", 
       value.name = "Unit_Sales")

## Calculating the number of NA values for each column in each dataset
data.frame(calendar = colSums(is.na(calendar)))
data.frame(sell_prices = colSums(is.na(sell_prices)))
data.frame(train = colSums(is.na(train)))

## Calculating the missing values for sell_price column 
zero_sum = function(x) sum(x = 0)

data.frame(sell_price = zero_sum(sell_prices$sell_price), Unit_Sales = zero_sum(train$Unit_Sales), row.names = "# of zero values")

## Adding additional features
sell_prices <- sell_prices %>%

  ## Creating a unique id based on the following combinations: item, store, time
  mutate(id = paste0(item_id, "_", store_id, "_validation_", wm_yr_wk))

sell_prices <- sell_prices %>%
  group_by(item_id, store_id, wm_yr_wk) %>%
  mutate(price_change = (sell_price - lag(sell_price)),
         price_cat = ifelse(sell_price <= 5, "5 USD or less",
                            ifelse(sell_price <= 10, "5.01 to 10 USD",
                                   ifelse(sell_price <= 15, "10.01 to 15 USD",
                                          ifelse(sell_price <= 25, "15.01 to 25 USD", "More than 25 USD"))))
  ) %>%
  select(id, price_cat, sell_price, price_change) ## Selecting only relevant features


## Merging the calendar data into the train data based on the column d
train <- left_join(train, calendar,
                   by = c("d" = "d")) 

## Adding the date into the train id to make the train id unique date-wise
train$id <- paste0(train$id, "_", train$wm_yr_wk)

## Selecting the relevant columns for this test
train <- train %>%
  select(d, id, item_id, dept_id, cat_id, store_id, state_id, Unit_Sales, date, weekday, month, year, event_name_1, event_name_2)






## Stratified Sampling based on the following categories: location, store, product category, and department
set.seed(100)
train <- stratified(train, c("year", "cat_id", "state_id"), 10000)

## Merging the sell_prices data into the train data based on the column wm_yr_wk
train <- left_join(train, sell_prices, 
                   by = c("id" = "id"))

#####################################################################























#####################################################################
## Render & Forward defined results to RMarkdown File
#####################################################################
library(rmarkdown)
library(knitr)
render("main.Rmd",
      output_format = "html_document",
      output_file = "Capstone.html")
