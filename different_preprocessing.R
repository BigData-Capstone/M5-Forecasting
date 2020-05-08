#importing relevant libraries
library(tidyverse)
library(data.table)
library(RcppRoll)
library(dplyr)
library(janitor)
library(forecast)
train <- fread("sales_train_validation.csv", stringsAsFactors = FALSE)
calendar <- fread("calendar.csv", stringsAsFactors = TRUE)
free <- function() invisible(gc())
View(calendar)


train <- train %>% 
  #eliminate valitrainion to save memory
  mutate(id = gsub("_validation", "", id)) %>%
  select(-item_id) %>% 
  select(-dept_id) %>%
  select(-cat_id) %>%
  select(-store_id) %>%
  select(-state_id) %>%
  mutate_if(is.factor, as.integer)
  

#clear the memory
free()

#transpose the data
train = t(train)
free()

#write first row as column header
train <- train %>%
  row_to_names(row_number = 1)



#create test dataframe
new_train = train[,9]

new_train <- as.numeric(new_train)
new_train <- as.data.frame(new_train)
str(new_train)
#add features
new_train$date = calendar$date[1:1913]
new_train$day = calendar$weekday[1:1913]
new_train$week = week(new_train$date)

#eliminate first 6 entries and last 4 entries to have full weeks only
new_train = new_train[-c(1:6, 1910:1913),]

#rename column
names(new_train)[1] <- "quantity"
View(new_train)



#create time series data
product_ts = ts(new_train$quantity, frequency = 7)
plot(product_ts)

### Create Test and Training Data
######################################################
index = tail(1:nrow(new_train),28)
train_product = new_train[-index,]
test_product = new_train[index,]

str(train_product)
### Naive Forecast & Evaluation Workbench
###########################################

train_product_ts = ts(train_product$quantity, frequency = 7)
forecast = naive(train_product_ts, h = 7)
tail(train_product$new_train,1)

autoplot(forecast, PI = FALSE)

# RMSE calculation
RMSE=function(actual, predicted){
  rmse = sqrt(mean((actual-predicted)^2))
  return(rmse)
}

View(test_product)
actual = test_product[13,,]$quantity
actual

for (w in unique(test_product$week)){
  
  
}


naive_error = walkforward_evaluation(train_product, test_product)
naive_error
mean(naive_error)



