#importing relevant libraries
library(tidyverse)
library(data.table)
library(RcppRoll)
devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
library(catboost)

library(lightgbm)

#garbage collector for training as a lot of memory is used
free <- function() invisible(gc())

# Create Constants for prediction
# First day to start prediction
FIRST_PREDICTION_DAY <- 1914 
# Predictions need to be made for 28 days
LENGTH_PREDICTION <- 28  

#Import the data
calendar <- fread("calendar.csv", stringsAsFactors = TRUE, 
                  drop = c("date", "weekday", "event_type_1", "event_type_2"))
#drop first 1000 for conserving memory
train <- fread("sales_train_validation.csv", stringsAsFactors = TRUE, drop = paste0("d_", 1:1000))
prices <- fread("sell_prices.csv", stringsAsFactors = TRUE)
                
#show dimensions
dim(train)



#create additional features
# Functions
#what does it do?
d2int <- function(X) {
  X %>% extract(d, into = "d", "([0-9]+)", convert = TRUE)
}

#Create additional features for the demand
demand_features <- function(X) {
  X %>% 
    group_by(id) %>% 
    mutate(lag_7 = dplyr::lag(demand, 7),
           lag_28 = dplyr::lag(demand, 28),
           roll_lag7_w7 = roll_meanr(lag_7, 7),
           roll_lag7_w28 = roll_meanr(lag_7, 28),
           roll_lag28_w7 = roll_meanr(lag_28, 7),
           roll_lag28_w28 = roll_meanr(lag_28, 28)) %>% 
    ungroup() 
}

#decrease memory by removing unused parts
free()

# Fill in blank values for prediction after the last day of data
train[, paste0("d_", FIRST_PREDICTION_DAY:(FIRST_PREDICTION_DAY + 2 * LENGTH_PREDICTION - 1))] <- NA
free()
train <- train %>% 
  #eliminate validation to save memory
  mutate(id = gsub("_validation", "", id)) %>% 
  #transpose the data to create a better overview
  gather("d", "demand", -id, -item_id, -dept_id, -cat_id, -store_id, -state_id) %>% 
  #what does it do?
  d2int() %>% 
  #merge the dataset
  left_join(calendar %>% d2int(), by = "d") %>% 
  left_join(prices, by = c("store_id", "item_id", "wm_yr_wk"))  %>%
  # what does it do?
  select(-wm_yr_wk) %>% 
  #convert various variables
  mutate(demand = as.numeric(demand)) %>% 
  mutate_if(is.factor, as.integer) %>% 
  #calculate lag
  demand_features() %>% 
  #lag cannot calculated for the last variables
  filter(d >= FIRST_PREDICTION_DAY | !is.na(roll_lag28_w28))
  

head(train,100)
# Response and features
y <- "demand"
x <- setdiff(colnames(train), c(y, "d", "id"))


#create test and training set
test <- train %>% 
  filter(d >= FIRST_PREDICTION_DAY - 56)

train <- train %>% 
  filter(d < FIRST_PREDICTION_DAY)

set.seed(3134)
idx <- sample(nrow(train), trunc(0.1 * nrow(train)))
valid <- catboost.load_pool(train[idx, x], label = train[[y]][idx])
train <- catboost.load_pool(train[-idx, x], label = train[[y]][-idx])
rm(prices, idx, calendar)

#release memory
free()

# Parameters
params <- list(iterations = 2000,
               metric_period = 100,
               #       task_type = "GPU",
               loss_function = "RMSE",
               eval_metric = "RMSE",
               random_strength = 0.5,
               depth = 7,
               # early_stopping_rounds = 400,
               learning_rate = 0.2,
               l2_leaf_reg = 0.1,
               random_seed = 93)

# Fit
fit <- catboost.train(train, valid, params = params)



                