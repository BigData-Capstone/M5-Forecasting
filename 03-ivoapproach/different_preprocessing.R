#importing relevant libraries
library(tidyverse)
library(data.table)
library(RcppRoll)
library(dplyr)
library(janitor)
library(forecast)
library(randomForest)
train <- fread("input/sales_train_validation.csv", stringsAsFactors = FALSE)
calendar <- fread("input/calendar.csv", stringsAsFactors = TRUE)
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


View(train)
#create test dataframe
new_train = train[,35]

new_train <- as.numeric(new_train)
new_train <- as.data.table(new_train)
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
tail(train_product$quantity,1)

autoplot(forecast, PI = FALSE)

# RMSE calculation
RMSE=function(actual, predicted){
  rmse = sqrt(mean((actual-predicted)^2))
  return(rmse)
}

# walkforward validation
walkforward_evaluation = function(train_product, test_product){
  
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

naive_error = walkforward_evaluation(train_product, test_product)
mean(naive_error)


save.image("~/R-Code/M5-Forecasting/03-ivoapproach/.RData")




### Seasonal Naive Forecast & Evaluation Workbench
###########################################

train_product_ts = ts(train_product$quantity, frequency = 7)
forecast = snaive(train_product_ts, h = 7)
tail(train_product$quantity,7)

autoplot(forecast, PI = FALSE)

# walkforward validation
walkforward_evaluation = function(train_product, test_product){
  
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

snaive_error = walkforward_evaluation(train_product, test_product)
mean(snaive_error)




### ARIMA
######################################################

# ACF Plot
ggAcf(train_product_ts)

# PACF Plot
ggPacf(train_product_ts)

# AR1 model
ar1 = arima(train_product_ts, order=c(1,0,0))
summary(ar1)
forecast(ar1, train_product_ts, h = 7)$mean

# ARMA11 model
arma11 = arima(train_product_ts, order=c(1,0,1))
summary(arma11)
forecast(arma11, train_product_ts, h = 7)$mean

# Auto Arima
autoarima = auto.arima(train_product_ts)
summary(autoarima)

# walkforward validation
walkforward_evaluation = function(model, train_product, test_product){
  
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

arima_error = walkforward_evaluation(autoarima, train_product, test_product)
mean(arima_error)










### Going Machine Learning
#######################################

# basic data prep
train_product$date=NULL
test_product$date=NULL

train_product$day=as.numeric(as.factor(train_product$day))
test_product$day=as.numeric(as.factor(test_product$day))


# create lags
train_product$quantity_lag7 = shift(train_product$quantity, n=7, fill=NA, type="lag")
train_product$quantity_lag8 = shift(train_product$quantity, n=8, fill=NA, type="lag")
train_product$quantity_lag9 = shift(train_product$quantity, n=9, fill=NA, type="lag")
train_product$quantity_lag10 = shift(train_product$quantity, n=10, fill=NA, type="lag")
train_product$quantity_lag11 = shift(train_product$quantity, n=11, fill=NA, type="lag")
train_product$quantity_lag12 = shift(train_product$quantity, n=12, fill=NA, type="lag")
train_product$quantity_lag13 = shift(train_product$quantity, n=13, fill=NA, type="lag")
train_product$quantity_lag14 = shift(train_product$quantity, n=14, fill=NA, type="lag")
View(train_product)

# decompose time series
train_product_decomposed = decompose(ts(train_product$quantity, frequency = 7))
train_product$seasonal = train_product_decomposed$seasonal

str(train_product)

train_product = train_product[complete.cases(train_product),]

tsforest = randomForest(quantity ~ ., train_product, ntree=10000)


# walkforward validation
walkforward_evaluation = function(model, train_product, test_product){
  
  history = train_product
  performance_collector = c()
  
  for (w in unique(test_product$week)){
    
    dat = tail(history, 14)
    dat = rbind(dat, test_product[week == w,,], fill=TRUE)
    
    # create lags
    dat$quantity_lag7 = shift(dat$quantity, n=7, fill=NA, type="lag")
    dat$quantity_lag8 = shift(dat$quantity, n=8, fill=NA, type="lag")
    dat$quantity_lag9 = shift(dat$quantity, n=9, fill=NA, type="lag")
    dat$quantity_lag10 = shift(dat$quantity, n=10, fill=NA, type="lag")
    dat$quantity_lag11 = shift(dat$quantity, n=11, fill=NA, type="lag")
    dat$quantity_lag12 = shift(dat$quantity, n=12, fill=NA, type="lag")
    dat$quantity_lag13 = shift(dat$quantity, n=13, fill=NA, type="lag")
    dat$quantity_lag14 = shift(dat$quantity, n=14, fill=NA, type="lag")
    
    # decompose time series
    dat$seasonal[15:21]=dat$seasonal[1:7]
    
    dat=tail(dat,7)
    
    #make forecast and assess RMSE
    predicted = predict(tsforest, dat)
    actual = test_product[week == w,,]$quantity
    performance = RMSE(actual, predicted)
    
    # update history and collect performance
    history = rbind(history, dat)
    performance_collector = c(performance_collector, performance)
  }
  
  return(performance_collector)
}

ml_error = walkforward_evaluation(tsforest, train_product, test_product)
mean(ml_error)

varImpPlot(tsforest)


# light gbm
