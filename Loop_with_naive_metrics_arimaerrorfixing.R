#importing relevant libraries
library(tidyverse)
library(data.table)
library(RcppRoll)
library(dplyr)
library(janitor)
library(forecast)
library(foreach)

setDTthreads(12)


#Definition of all functions

# RMSE calculation
RMSE=function(actual, predicted){
  rmse = sqrt(mean((actual-predicted)^2))
  return(rmse)
}
# walkforward validation naive
walkforward_evaluation_naive = function(train_product, test_product){
  
  history = train_product
  performance_collector = c()
  
  for (w in unique(test_product$week)){
    #create time series of history
    history_ts = ts(history$quantity, frequency = 7)
    
    #make forecast and assess RMSE
    predicted = naive(history_ts, h = 7)$mean
    actual = test_product[week == w,,]$quantity
    performance = RMSE(actual, predicted)
    
    # update history and collect performance
    history = rbind(history, test_product[week == w,,])
    performance_collector = c(performance_collector, performance)
  }
  
  return(performance_collector)
}

# walkforward validation snaiv
walkforward_evaluation_snaive = function(train_product, test_product){
  
  history = train_product
  performance_collector = c()
  
  for (w in unique(test_product$week)){
    #create time series of history
    history_ts = ts(history$quantity, frequency = 7)
    
    #make forecast and assess RMSE
    predicted = snaive(history_ts, h = 7)$mean
    actual = test_product[week == w,,]$quantity
    performance = RMSE(actual, predicted)
    
    # update history and collect performance
    history = rbind(history, test_product[week == w,,])
    performance_collector = c(performance_collector, performance)
  }
  
  return(performance_collector)
}

# walkforward validation autoarima
walkforward_evaluation_autoarima = function(model, train_product, test_product){
  
  history = train_product
  performance_collector = c()
  
  for (w in unique(test_product$week)){
    #create time series of history
    history_ts = ts(history$quantity, frequency = 7)
    
    #make forecast and assess RMSE
    predicted = forecast(model, history_ts, h = 7)$mean
    actual = test_product[week == w,,]$quantity
    performance = RMSE(actual, predicted)
    
    # update history and collect performance
    history = rbind(history, test_product[week == w,,])
    performance_collector = c(performance_collector, performance)
  }
  
  return(performance_collector)
}

#define function for clearing memory
free <- function() invisible(gc())

#read in the data
train <- fread("sales_train_validation.csv", stringsAsFactors = FALSE)
calendar <- fread("calendar.csv", stringsAsFactors = TRUE)


#do some data wrangling
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
# View(calendar)
#write first row as column header
train <- train %>%
  row_to_names(row_number = 1)

#eliminate last two rows to have full weeks only and first 1500 to save memory
train = train[-c(1912:1913),]
train = train[-c(1:1505),]

#change the week notation in the calendar
calendar$wm_yr_wk = as.character(calendar$wm_yr_wk)

#elimnate all first three characters to get the week number
calendar$wm_yr_wk <- gsub("^.{0,3}", "", calendar$wm_yr_wk)

#convert to integer
calendar$wm_yr_wk = as.integer(calendar$wm_yr_wk)


#preparation for looping through the data
iterations = 5000
variables = 3

results_matrix <- matrix(ncol=variables, nrow=iterations)



computing_start_time <- Sys.time()


i=1
while(i <= iterations){
  iteration_start_time <- Sys.time()
  #create dataframe for individual product
  new_train = train[,i]
  #convert data
  new_train <- as.numeric(new_train)
  new_train <- as.data.table(new_train)
  #add features
  new_train$date = calendar$date[1506:1911]
  new_train$day = calendar$weekday[1506:1911]
  new_train$week = calendar$wm_yr_wk[1506:1911]
  #rename column
  names(new_train)[1] <- "quantity"
  
  # Create Test and Training Data
  index = tail(1:nrow(new_train),28)
  train_product = new_train[-index,]
  test_product = new_train[index,]
  train_product_ts = ts(train_product$quantity, frequency = 7)
  
  # Naive Forecast & Evaluation Workbench
  naive_error = walkforward_evaluation_naive(train_product, test_product)
  results_matrix[i,1] = mean(naive_error)
  
  # Seasonal Naive Forecast & Evaluation Workbench
  snaive_error = walkforward_evaluation_snaive(train_product, test_product)
  results_matrix[i,2] = mean(snaive_error)
  
  # ARIMA
  #only try, as there is sometimes an error (like element 340)
  try({
    autoarima = auto.arima(train_product_ts)
    arima_error = walkforward_evaluation_autoarima(autoarima, train_product, test_product)
    results_matrix[i,3] = mean(arima_error)
  }, silent = FALSE)
  
  
  #free memory
  free()
  #give an overview of progress
  iteration_end_time <- Sys.time()
  iteration_time <- iteration_end_time - iteration_start_time
  print(c(i,iteration_time))
  #increment i
  i=i+1
}


computing_end_time <- Sys.time()
computing_time <- computing_end_time - computing_start_time
print(c(computing_time,computing_time/iterations))


View(results_matrix[])
mean(results_matrix[,1])
mean(results_matrix[,2])
mean(results_matrix[,3])
sum(is.na(results_matrix[,3]))

