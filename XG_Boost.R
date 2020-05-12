#importing relevant libraries
library(tidyverse)
library(data.table)
library(RcppRoll)
library(dplyr)
library(janitor)
library(forecast)
library(xgboost)
library(Matrix)
library(mltools)
library(caret)
library(foreach)
library(doParallel)

#define function for clearing memory
free <- function() invisible(gc())

#read in the data 
sales_train_validation.csv <- fread("sales_train_validation.csv", stringsAsFactors = TRUE)
calendar.csv <- fread("calendar.csv", stringsAsFactors = TRUE)
sell_prices.csv <- fread("sell_prices.csv", stringsAsFactors = TRUE)



#create the dataset
dataset <- data.table::melt.data.table(
  data = sales_train_validation.csv,
  id.vars = colnames(sales_train_validation.csv)[grepl("id", colnames(sales_train_validation.csv))],
  measure.vars = colnames(sales_train_validation.csv)[grepl("^d_", colnames(sales_train_validation.csv))],
  variable.name = "d",
  value.name = "demand",
  na.rm = FALSE
)

#remove csv and clear memory
rm(sales_train_validation.csv)
free()

# make data smaller
dataset[, ("d") := as.integer(gsub("^d_", "", get("d")))]
free()

# ensure order
data.table::setorder(
  x = dataset,
  cols = "d"
)
free()


# define indices for prediction/evaluation, training and testing set
train_index <- 1431 # use the last 1.5 yrs for training -> Full 2015 + Half 2016
test_index <- (1913-28) #predict the last 28 days

# reduce data
dataset <- dataset[get("d") >= train_index, ]
free()


#Make sure the dataset is sorted correctly
stopifnot(!is.unsorted(dataset$d))

#merge the calendar
calendar.csv[, `:=` (weekend = ifelse(get("weekday") %in% c("Saturday", "Sunday"), 1L, 0L),
                     # create weekend feature
                     d = as.integer(gsub("^d_", "", get("d"))),
                     day = as.integer(substr(get("date"), start = 9, stop = 10)),
                     date = factor(as.Date(get("date"))),
                     event_name_1 = as.integer(factor(get("event_name_1"))),
                     event_type_1 = as.integer(factor(get("event_type_1"))),
                     event_name_2 = as.integer(factor(get("event_name_2"))),
                     event_type_2 = as.integer(factor(get("event_type_2"))))][
                       , `:=` (date = NULL,
                               weekday = NULL)
                       ]

# merge calendar to dataset
dataset <- calendar.csv[get("d") >= train_index, ][dataset, on = "d"]
rm(calendar.csv)
free()


# merge prices to dataset
dataset <- sell_prices.csv[dataset, on = c("store_id", "item_id", "wm_yr_wk")][, wm_yr_wk := NULL]
rm(sell_prices.csv)
free()

# create more features (slightly modified code from https://www.kaggle.com/mayer79/m5-forecast-attack-of-the-data-table)
agg <- dataset[, .(agg_mean = mean(demand, na.rm = TRUE)), keyby = c("item_id", "d")]
agg[, lag_t28 := dplyr::lag(agg_mean, 28), keyby = "item_id"
    ][
      , rolling_mean_t30r := RcppRoll::roll_meanr(lag_t28, 30), keyby = "item_id"
      ][
        , rolling_sd_t30r := RcppRoll::roll_sdr(lag_t28, 30), keyby = "item_id"
        ]
#Merge features to dataset
dataset[agg, `:=` (rolling_mean_t30r_item = i.rolling_mean_t30r,
                   rolling_sd_t30r_item = i.rolling_sd_t30r
                  ), on = c("item_id", "d")]

demand_features <- function(X) {
  X %>% 
    group_by(id) %>% 
    mutate(
      
      lag_1 = dplyr::lag(demand, 1),
      lag_2 = dplyr::lag(demand, 2),
      lag_3 = dplyr::lag(demand, 3),
      lag_7 = dplyr::lag(demand, 7),
      lag_28 = dplyr::lag(demand, 28),
      roll_lag7_w7 = roll_meanr(lag_7, 7),
      roll_lag7_w28 = roll_meanr(lag_7, 28),
      roll_lag28_w7 = roll_meanr(lag_28, 7),
      roll_lag28_w28 = roll_meanr(lag_28, 28)) %>% 
    ungroup() 
}

#more features
dataset <- dataset %>%
  demand_features()

str(dataset)
View(dataset)
rm(agg)
free()
test_df = filter(dataset, item_id == "FOODS_1_016")

View(test_df)
#filter only the items of store CA_3
dataset = filter(dataset, store_id == "CA_3")

#drop un-nessecary colums store id state id
dataset = select(dataset, -store_id)
dataset = select(dataset, -state_id)

#convert item id into numeric format
dataset$item_id = as.integer(dataset$item_id)
dataset$id = as.integer(dataset$id)


#Encode category and departement id as dummy variables
dataset$cat_id = one_hot(as.data.table(dataset$cat_id))
dataset$dept_id = one_hot(as.data.table(dataset$dept_id))

#clear memory
free()

#split the training data
train_dataset = filter(dataset, d <= test_index)
View(train_dataset)

test_dataset = filter(dataset, d > test_index)

#omit missing values
View(dataset)

#Assign label
train_label <- train_dataset$demand
test_label <- test_dataset$demand

View(x_train)

#remove label from dataset
train_dataset = select(train_dataset, -demand)
test_dataset = select(test_dataset, -demand)

#convert datasets to matrix
x_train = as.matrix(train_dataset)
x_test = as.matrix(test_dataset)



#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = x_train, label = train_label)


#set the parameter
params <- list(booster = "gbtree",
              objective = "reg:linear",
              eval_metric = "rmse",
              eta = 0.07,
              max_depth = 5,
              min_child_weight = 10,
              colsample_bytree = 1,
              gamma = 0.9,
              alpha = 1.0,
              subsample = 0.7
)

N_cpu = detectCores()
N_cpu
xgb.tab <- xgb.cv(data = trainDMatrix,
                  , param = params, evaluation = "rmse", nrounds = 100
                  , nthreads = N_cpu, nfold = 5, early_stopping_round = 10)


xgb.tab$best_iteration
model_xgb <- xgboost(data = trainDMatrix, param = params, nrounds = 10, importance = TRUE)


#predict smth.
View(x_test)
pred = predict(model_xgb, newdata = x_test)
View(pred)
View(test_label)
pred_salesData1 <- x_test %>%
  bind_cols(pred = predict(model_xgb, newdata = x_test)) %>%
  mutate(error = demand - pred)

#importance plot
importance <- xgb.importance(feature_names = colnames(trainMatrix), model = model)
xgb.ggplot.importance(importance_matrix = importance)

