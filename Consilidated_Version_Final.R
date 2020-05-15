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
library(ranger)
library(caret)

###helper functions
#define function for clearing memory
free <- function() invisible(gc())

#define mode function to find most frequent value
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

RMSE=function(actual, predicted){
  rmse = sqrt(mean((actual-predicted)^2))
  return(rmse)
}

#read in the data 
sales_train_validation.csv <- fread("sales_train_validation.csv", stringsAsFactors = TRUE)
calendar.csv <- fread("calendar.csv", stringsAsFactors = TRUE)
sell_prices.csv <- fread("sell_prices.csv", stringsAsFactors = TRUE)

###########################################################################################
### Creating simple dataset
###########################################################################################

#create simple dataset
#create simple (for naive, snaive and arima) 
#do some data wrangling
simple_dataset <- sales_train_validation.csv %>% 
  #eliminate valitrainion to save memory
  mutate(id = gsub("_validation", "", id)) %>%
  select(-item_id) %>% 
  select(-dept_id) %>%
  select(-cat_id) %>%
  #filter for specific store
  filter(store_id == "CA_3") %>%
  #eliminate further columns
  select(-store_id) %>%
  select(-state_id) %>%
  mutate_if(is.factor, as.integer)


#clear the memory
free()

#transpose the data
simple_dataset = t(simple_dataset)
free()

#write first row as column header
simple_dataset <- simple_dataset %>%
  row_to_names(row_number = 1)

#only leave last 1,5 years for training as the years before might not be relevant

simple_dataset = simple_dataset[-c(1:1430),]

#create training data for simple_dataset -> Last 28 days should be predicted
index = tail(1:nrow(simple_dataset),28)
train_simple = simple_dataset[-index,]
test_simple = simple_dataset[index,]

###########################################################################################
### Creating complex dataset
###########################################################################################

#create complex dataset
dataset <- data.table::melt.data.table(
  data = sales_train_validation.csv,
  id.vars = colnames(sales_train_validation.csv)[grepl("id", colnames(sales_train_validation.csv))],
  measure.vars = colnames(sales_train_validation.csv)[grepl("^d_", colnames(sales_train_validation.csv))],
  variable.name = "d",
  value.name = "demand",
  na.rm = FALSE
)

#remove csv and clear memory
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

#Make sure the dataset is sorted correctly
stopifnot(!is.unsorted(dataset$d))

# define indices for prediction/evaluation, training and testing set
train_index <- 1350 # use the last 1.5 yrs for training -> Full 2015 + Half 2016 use a little more data to avoid nas for lag
test_index <- (1913-28) #predict the last 28 days

# reduce data
dataset <- dataset[get("d") >= train_index, ]
free()

#merge the calendar
calendar.csv[, `:=` (weekend = ifelse(get("weekday") %in% c("Saturday", "Sunday"), 1L, 0L),
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
free()


# merge prices to dataset
dataset <- sell_prices.csv[dataset, on = c("store_id", "item_id", "wm_yr_wk")][, wm_yr_wk := NULL]
free()

# create more features
demand_features <- function(X) {
  X %>% 
    group_by(id) %>% 
    #create some lag variables 
    mutate(
      lag_1 = dplyr::lag(demand, 1),
      lag_2 = dplyr::lag(demand, 2),
      lag_3 = dplyr::lag(demand, 3),
      lag_4 = dplyr::lag(demand, 4),
      lag_5 = dplyr::lag(demand, 5),
      lag_6 = dplyr::lag(demand, 6),
      lag_7 = dplyr::lag(demand, 7),
      mean_last3 = (lag_1+lag_2+lag_3)/3,
      mean_last7 = (lag_1+lag_2+lag_3+lag_4+lag_5+lag_6+lag_7)/7,
      lag_14 = dplyr::lag(demand, 14),
      lag_21 = dplyr::lag(demand, 21),
      lag_28 = dplyr::lag(demand, 28)) %>% 
    ungroup()%>%
    
    #create more features regarding the demand
    group_by(item_id)%>%
    mutate(
      demand_min = min(demand, na.rm = FALSE),
      demand_mean = mean(demand, na.rm = FALSE),
      demand_median = median(demand, na.rm = FALSE),
      demand_max = max(demand, na.rm = FALSE),
      demand_mode = Mode(demand),
      demand_sd = sd(demand, na.rm = FALSE)
    )%>%
    ungroup()
  
  
}

#create features before reducing dataset to avoid NA's
dataset <- dataset %>%
  demand_features()

#clear memory and remove csv's
rm(sales_train_validation.csv)
rm(calendar.csv)
rm(sell_prices.csv)
free()

#filter only the items of store CA_3
dataset = filter(dataset, store_id == "CA_3")

#drop un-nessecary colums store_id, state id, cat_id, dept_id & id
dataset = select(dataset, -store_id)
dataset = select(dataset, -state_id)
dataset = select(dataset, -cat_id)
dataset = select(dataset, -dept_id)
dataset = select(dataset, -id)

#clear memory again
free()

#convert item id of the dataset into numeric format (starts at 1438)
dataset$item_id = as.integer(dataset$item_id)

#create training for the more complex dataset
#create a list of all the items
item_id_df <- data.frame(dataset$item_id)
item_id_df <- unique(item_id_df)

#split the training data
#split the training data
train_dataset = filter(dataset, d >= 1431 & d < 1886)
intermediate_test_dataset_1 = filter(dataset, d >= 1800 & d <= 1885)
intermediate_test_dataset_2 = filter(dataset, d == 1886)
test_dataset = filter(dataset, d > 1886)

#clear memory and maybe remove dataset
free()

#remove demand and all the lag metrics from the testset as they would not be available with the exception of the first day (1886)
test_dataset[,17:34] <-NA
intermediate_test_dataset_1 = rbind(intermediate_test_dataset_1,intermediate_test_dataset_2, test_dataset)

#rename complete dataset
complete_test_dataset = intermediate_test_dataset_1


###########################################################################################
### XGBoost Implementation 
###########################################################################################
##select item from item list
item_id = item_id_df[1,1]

#filter for this variable to get the train set 
subset_train = filter(train_dataset, item_id == !!item_id)
subset_train = as.data.frame(subset_train)

#Assign label
train_label = subset_train$demand

#remove label from dataset
xg_train_dataset = select(subset_train, -demand)

#convert datasets to matrix
x_train = as.matrix(xg_train_dataset)

#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = x_train, label = train_label)

#set the parameter
params <- list(booster = "gbtree",
               objective = "reg:linear",
               eval_metric = "rmse",
               eta = 0.2,
               max_depth = 1,
               min_child_weight = 10,
               colsample_bytree = 1,
               gamma = 0,
               alpha = 1.0,
               subsample = 0.7
)

#detect the number of cores for multicore operation
N_cpu = detectCores()

#find the number of iterations to build the best model
xgb.tab <- xgb.cv(data=trainDMatrix, param = params, evaluation = "rmse", nrounds = 100
                  , nthreads = N_cpu, nfold = 5, early_stopping_round = 10)


#build the model
model_xgb <- xgboost(data = trainDMatrix, param = params, nrounds = xgb.tab$best_iteration, importance = TRUE)



###########################################################################################
### Looping through the data
###########################################################################################

#preparation for looping through the data
#iterations = ncol(train_simple)
variables = 5
results_matrix <- matrix(ncol=variables, nrow=10)
computing_start_time <- Sys.time()



i=1
#loop through the data
for (i in 1:10){
  #########################################################################################
  ### Data Preparation for simple approaches
  #########################################################################################
  iteration_start_time <- Sys.time()
  #create training data frame
  new_train = train_simple[,i]
  #convert data
  new_train <- as.numeric(new_train)
  new_train <- as.data.frame(new_train)
  #rename column
  names(new_train)[1] <- "quantity"
  
  #create dataframe for test
  new_test = test_simple[,i]
  new_test <- as.numeric(new_test)
  new_test <- as.data.frame(new_test)
  names(new_test)[1] <- "quantity"
  
  # Create time series data
  train_product_ts = ts(new_train$quantity, frequency = 7)
  
  #########################################################################################
  ### Naive Forecast
  #########################################################################################
  
  # Naive Forecast 
  #make forecast and assess RMSE
  predicted = naive(train_product_ts, h = 28)$mean
  
  #convert to df for further calculations
  predicted_df = as.data.frame(predicted)
  
  #Create performance collector object to store rmsw values for every day
  performance_collector <- matrix(ncol=1, nrow=28)
  
  #calculate rmse 
  actual = new_test[,1]
  predicted = predicted_df[,1]
  performance = RMSE(actual, predicted)
  
  #write performance into results matrix
  results_matrix[i,1] = performance
  
  #########################################################################################
  ### SNaive Forecast
  #########################################################################################
  
  
  #### SNaive Forecast 
  #make forecast and assess RMSE
  snaive_predicted = snaive(train_product_ts, h = 28)$mean
  
  #convert to df for further calculations
  snaive_predicted_df = as.data.frame(snaive_predicted)
  
  #Create performance collector object to store rmsw values for every day
  performance_collector <- matrix(ncol=1, nrow=28)
  
  #calculate rmse 
  actual = new_test[,1]
  snaive_predicted = snaive_predicted_df[,1]
  performance = RMSE(actual, snaive_predicted)
  
  #write results into results matrix
  results_matrix[i,2] = performance
  
  #########################################################################################
  ### Arima Forecast
  #########################################################################################
  
  #only try, as there is sometimes an error (like element 340)
  try({
    autoarima = auto.arima(train_product_ts)
    autoarima_predicted = forecast(autoarima, train_product_ts, h = 28)$mean
    
    #convert to df for further calculations
    autoarima_predicted_df = as.data.frame(autoarima_predicted)
    
    #Create performance collector object to store rmsw values for every day
    performance_collector <- matrix(ncol=1, nrow=28)
    
    #loop to calculate rmse for every day
    actual = new_test[,1]
    autoarima_predicted = autoarima_predicted_df[,1]
    performance = RMSE(actual, autoarima_predicted)
    
    #write results into results matrix
    results_matrix[i,3] = performance
  }, silent = FALSE)
  
  
  #########################################################################################
  ### Random Forest Forecast
  #########################################################################################
  ##select item from item list
  item_id = item_id_df[i,1]
  
  #filter for the item variable to get the train set 
  subset_train = filter(train_dataset, item_id == !!item_id)
  subset_train_df = as.data.frame(subset_train)
  
  #remove price to avoid missing values
  subset_train_df = select(subset_train_df,-sell_price)
  
  #get number of cpus
  N_cpu = detectCores()
  
  #create model
  rangermodel = ranger(formula = demand~ ., data=subset_train_df, num.trees = 1000, num.threads = N_cpu)
  
  #initialize predictions matrix to store individual predictions
  predictions_matrix_rf = matrix(nrow = 28, ncol = 1)
  
  #filter for the item variable to get the test set 
  subset_test = filter(complete_test_dataset, item_id == !!item_id)
  
  #convert to df
  subset_test_df = as.data.frame(subset_test)
  
  #remove price to avoid missing values
  subset_test_df = select(subset_test_df,-sell_price)
  
  #create label for the last 28 predictions
  subset_test_label = tail(subset_test_df$demand,28)
  
  #Fill the last 28 predictions with missing values
  subset_test_df[87:114,]$demand <- NA
  
  #fill the statistics of the demand into the dataframe
  subset_test_df[87:114,]$demand_min = subset_test_df[86,]$demand_min
  subset_test_df[87:114,]$demand_max = subset_test_df[86,]$demand_max
  subset_test_df[87:114,]$demand_median = subset_test_df[86,]$demand_median
  subset_test_df[87:114,]$demand_mean = subset_test_df[86,]$demand_mean
  subset_test_df[87:114,]$demand_sd = subset_test_df[86,]$demand_sd
  subset_test_df[87:114,]$demand_mode = subset_test_df[86,]$demand_mode
  
  #predict in loop -> Better description needed
  x = 1
  for (x in 1:28) {
    # 1885 is the last training day
    counter = 1885 + x  
    
    #only filter one day for the testset
    test_1_day = filter(subset_test_df, d == !!counter)
    
    #predict -> mit test_1 
    pred_rf = predict(rangermodel,data = test_1_day)
    #schreibe es in predictions matrix
    predictions_matrix_rf[x,1] = pred_rf$predictions
    #update dataframe with predicted demand
    current_day = 86 + x #86 is the row last training day
    
    #substitution for prediction -> Demand is in column 15
    subset_test_df[current_day,15] = pred_rf$predictions

    #select the next day to calculate lags
    next_day = current_day +1
    
    if(next_day < 115){
      #calculate lag for next day
      subset_test_df[next_day,]$lag_1 = subset_test_df[(next_day-1),]$demand
      subset_test_df[next_day,]$lag_2 = subset_test_df[(next_day-2),]$demand
      subset_test_df[next_day,]$lag_3 = subset_test_df[(next_day-3),]$demand
      subset_test_df[next_day,]$lag_4 = subset_test_df[(next_day-4),]$demand
      subset_test_df[next_day,]$lag_5 = subset_test_df[(next_day-5),]$demand
      subset_test_df[next_day,]$lag_6 = subset_test_df[(next_day-6),]$demand
      subset_test_df[next_day,]$lag_7 = subset_test_df[(next_day-7),]$demand
      subset_test_df[next_day,]$lag_14 = subset_test_df[(next_day-14),]$demand
      subset_test_df[next_day,]$lag_21 = subset_test_df[(next_day-21),]$demand
      subset_test_df[next_day,]$lag_28 = subset_test_df[(next_day-28),]$demand
      
      #calculate means
      subset_test_df[next_day,]$mean_last3 = (subset_test_df[next_day,]$lag_1 + subset_test_df[next_day,]$lag_2 + subset_test_df[next_day,]$lag_3)/3
      subset_test_df[next_day,]$mean_last7 = (subset_test_df[next_day,]$lag_1 + 
                                                subset_test_df[next_day,]$lag_2 + 
                                                subset_test_df[next_day,]$lag_3 +
                                                subset_test_df[next_day,]$lag_4 + 
                                                subset_test_df[next_day,]$lag_5 + 
                                                subset_test_df[next_day,]$lag_6 +
                                                subset_test_df[next_day,]$lag_7)/3
    }
    
    
  }
  
  
  
  #calculate RMSE
  actual = as.matrix(subset_test_label)
  predicted = predictions_matrix_rf
  performance = RMSE(actual, predicted)
  
  #write results into results matrix
  results_matrix[i,4] = performance
  
  #########################################################################################
  ### ML Approach Forecast: XGboost 
  #########################################################################################
  ##select item from item list
  item_id = item_id_df[i,1]
  
  #filter for this variable to get the train set 
  subset_train = filter(train_dataset, item_id == !!item_id)
  subset_train = as.data.frame(subset_train)
  
  #Assign label
  train_label = subset_train$demand
  
  #remove label from dataset
  xg_train_dataset = select(subset_train, -demand)
  
  #convert datasets to matrix
  x_train = as.matrix(xg_train_dataset)

  #Create input for xgboost
  trainDMatrix <- xgb.DMatrix(data = x_train, label = train_label)
  
  #set the parameter
  params <- list(booster = "gbtree",
                 objective = "reg:linear",
                 eval_metric = "rmse",
                 eta = 0.1,
                 max_depth = 1,
                 min_child_weight = 10,
                 colsample_bytree = 1,
                 gamma = 0,
                 alpha = 1.0,
                 subsample = 0.7
  )
  
  #detect the number of cores for multicore operation
  N_cpu = detectCores()
  
  xgb.tab <- xgb.cv(data=trainDMatrix, param = params, evaluation = "rmse", nrounds = 100
                    , nthreads = N_cpu, nfold = 5, early_stopping_round = 10)
  
  #build the model
  model_xgb <- xgboost(data = trainDMatrix, param = params, nrounds = xgb.tab$best_iteration , importance = TRUE)
  
  #initialize predictions matrix to store individual predictions
  predictions_matrix_xgb = matrix(nrow = 28, ncol = 1)
  
  #filter for the item variable to get the test set 
  subset_test = filter(complete_test_dataset, item_id == !!item_id)
  
  #convert to df
  subset_test_df = as.data.frame(subset_test)
  
  #create label for the last 28 predictions
  subset_test_label = tail(subset_test_df$demand,28)
  
  #Fill the last 28 predictions with missing values
  subset_test_df[87:114,]$demand <- NA
  
  #fill the statistics of the demand into the dataframe
  subset_test_df[87:114,]$demand_min = subset_test_df[86,]$demand_min
  subset_test_df[87:114,]$demand_max = subset_test_df[86,]$demand_max
  subset_test_df[87:114,]$demand_median = subset_test_df[86,]$demand_median
  subset_test_df[87:114,]$demand_mean = subset_test_df[86,]$demand_mean
  subset_test_df[87:114,]$demand_sd = subset_test_df[86,]$demand_sd
  subset_test_df[87:114,]$demand_mode = subset_test_df[86,]$demand_mode
  
  #predict in loop -> Better description needed
  x = 1
  for (x in 1:28) {
    # 1885 is the last training day
    counter = 1885 + x  
    
    #only filter one day for the testset
    test_1_day = filter(subset_test_df, d == !!counter)
    
    #remove demand column
    test_1_day = select(test_1_day, -demand)
 
    #convert to matrix 
    test_1_day_matrix = as.matrix(test_1_day)

    #predict -> mit test_1 
    xgb_predicted = predict(model_xgb, newdata = test_1_day_matrix)
    
    #convert to matrix
    xgb_predicted_result= as.matrix(xgb_predicted)
    
    #schreibe es in predictions matrix
    predictions_matrix_xgb[x,1] = xgb_predicted_result
    
    #update dataframe with predicted demand
    current_day = 86 + x #86 is the row last training day
    
    #substitution for prediction -> Demand is in column 15
    subset_test_df[current_day,15] = xgb_predicted_result
    
    #select the next day to calculate lags
    next_day = current_day +1
    
    if(next_day < 115){
      #calculate lag for next day
      subset_test_df[next_day,]$lag_1 = subset_test_df[(next_day-1),]$demand
      subset_test_df[next_day,]$lag_2 = subset_test_df[(next_day-2),]$demand
      subset_test_df[next_day,]$lag_3 = subset_test_df[(next_day-3),]$demand
      subset_test_df[next_day,]$lag_4 = subset_test_df[(next_day-4),]$demand
      subset_test_df[next_day,]$lag_5 = subset_test_df[(next_day-5),]$demand
      subset_test_df[next_day,]$lag_6 = subset_test_df[(next_day-6),]$demand
      subset_test_df[next_day,]$lag_7 = subset_test_df[(next_day-7),]$demand
      subset_test_df[next_day,]$lag_14 = subset_test_df[(next_day-14),]$demand
      subset_test_df[next_day,]$lag_21 = subset_test_df[(next_day-21),]$demand
      subset_test_df[next_day,]$lag_28 = subset_test_df[(next_day-28),]$demand
      
      #calculate means
      subset_test_df[next_day,]$mean_last3 = (subset_test_df[next_day,]$lag_1 + subset_test_df[next_day,]$lag_2 + subset_test_df[next_day,]$lag_3)/3
      subset_test_df[next_day,]$mean_last7 = (subset_test_df[next_day,]$lag_1 + 
                                                subset_test_df[next_day,]$lag_2 + 
                                                subset_test_df[next_day,]$lag_3 +
                                                subset_test_df[next_day,]$lag_4 + 
                                                subset_test_df[next_day,]$lag_5 + 
                                                subset_test_df[next_day,]$lag_6 +
                                                subset_test_df[next_day,]$lag_7)/3
    }
    
    
  }
  #Calculate rmse 
  actual = subset_test_label
  predicted = predictions_matrix_xgb
  performance = RMSE(actual, predicted)
  
  #write results into results matrix
  results_matrix[i,5] = performance
  
  #free memory
  free()
  #give an overview of progress
  iteration_end_time <- Sys.time()
  iteration_time <- iteration_end_time - iteration_start_time
  print(c(i,iteration_time))
}

#calculate mean of results (#Needs to be weighted depending on how much the product has been sold)
mean(results_matrix[1:10,1])
mean(results_matrix[1:10,2])
mean(results_matrix[1:10,3])
mean(results_matrix[1:10,4])
mean(results_matrix[1:10,5])

View(results_matrix)






