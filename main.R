## Capstone Project
# M5 - Forecasting
# Link zur Challenge: https://www.kaggle.com/c/m5-forecasting-accuracy

# to save the current RData to the .RData
save.image("~/R-Code/M5-Forecasting/.RData")

## Importing the Data
#####################################################################
library(readr)
calendar = read_csv("input/calendar.csv")
sales_train_validation = read_csv("input/sales_train_validation.csv")
sample_submission = read_csv("input/sample_submission.csv")
sell_prices = read_csv("input/sell_prices.csv")









#####################################################################
## Render & Forward defined results to RMarkdown File
#####################################################################
library(rmarkdown)
library(knitr)
render("main.Rmd",
      output_format = "html_document",
      output_file = "Capstone.html")
