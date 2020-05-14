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
# View(calendar)
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
      lag_28 = dplyr::lag(demand, 28),
      roll_lag7_w7 = roll_meanr(lag_7, 7),
      roll_lag7_w28 = roll_meanr(lag_7, 28),
      roll_lag28_w7 = roll_meanr(lag_28, 7),
      roll_lag28_w28 = roll_meanr(lag_28, 28)) %>% 
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
View(dataset)


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
train_dataset = filter(dataset, d >= 1431)
test_dataset = filter(dataset, d >= 1886)

#clear memory and maybe remove dataset
free()


###########################################################################################
### XGBoost Implementation 
###########################################################################################
#Assign label
train_label = train_dataset$demand

#remove label from dataset
xg_train_dataset = select(train_dataset, -demand)

#convert datasets to matrix
x_train = as.matrix(xg_train_dataset)

#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = x_train, label = train_label)

#set the parameter
params <- list(booster = "gbtree",
               objective = "reg:linear",
               eval_metric = "rmse",
               eta = 0.2,
               max_depth = 8,
               min_child_weight = 10,
               colsample_bytree = 1,
               gamma = 0,
               alpha = 1.0,
               subsample = 0.7
)

#detect the number of cores for multicore operation
N_cpu = detectCores()

#find the number of iterations to build the best model
xgb.tab <- xgb.cv(data=trainDMatrix, param = params, evaluation = "rmse", nrounds = 10
                  , nthreads = N_cpu, nfold = 5, early_stopping_round = 10)


#build the model
model_xgb <- xgboost(data = trainDMatrix, param = params, nrounds = xgb.tab$best_iteration, importance = TRUE)


###########################################################################################
### Catboost Implementation
###########################################################################################



###########################################################################################
### Looping through the data
###########################################################################################

#preparation for looping through the data
iterations = ncol(train_simple)
variables = 5
results_matrix <- matrix(ncol=variables, nrow=10)
computing_start_time <- Sys.time()
iterations = 10
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
  ### Data Preparation for ML approaches
  #########################################################################################
  #selecting the individual product
  #assign value of item _id array to temporary variable 
  
  #item_id = item_id[i,1]
  
  #filter for this variable 
  
  #subset_test = filter(train_dataset, item_id == !!item_id)
  
  #prediction step
  
  #loop to assess performance
  
  
  #########################################################################################
  ### Random Forest Forecast
  #########################################################################################
  ##select item from item list
  item_id = item_id_df[i,1]
  
  #filter for this variable to get the train set 
  subset_train = filter(train_dataset, item_id == !!item_id)
  subset_train = as.data.frame(subset_train)
  
  #remove price to avoid missing values
  subset_train = select(subset_train,-sell_price)
  
  #create model
  rangermodel = ranger(formula = demand~ ., data=subset_train, num.trees = 1000, num.threads = 4)

  #filter for the item to get the test set 
  subset_test = filter(test_dataset, item_id == !!item_id)
  subset_test = as.data.frame(subset_test)
  
  #remove price to avoid missing values
  subset_test = select(subset_test,-sell_price)
  
  #make prediction
  pred = predict(rangermodel,data = subset_test)
  pred_matrix = as.matrix(pred$predictions)
  
  #calculate RMSE
  actual = as.matrix(subset_test$demand)
  predicted = pred_matrix
  performance = RMSE(actual, predicted)
  
  #write results into results matrix
  results_matrix[i,4] = performance
  
  #########################################################################################
  ### ML Approach Forecast: XGboost -> Prediction only
  #########################################################################################
  ##select item from item list
  item_id = item_id_df[i,1]
  
  #filter for this variable 
  subset_test = filter(test_dataset, item_id == !!item_id)
  
  #get demand from dataset
  subset_test_label <- subset_test$demand
  subset_test_label_df <- as.data.frame(subset_test_label)
  
  #remove label from dataset
  subset_test = select(subset_test, -demand)
  
  #convert to matrix
  subset_test_matrix = as.matrix(subset_test)
  
  #make prediction
  xgb_predicted = predict(model_xgb, newdata = subset_test_matrix)
  #convert to df
  xgb_predicted_df = as.data.frame(xgb_predicted)
  
  #instantiate performance collector
  performance_collector <- matrix(ncol=1, nrow=28)
  
  #Calculate rmse 
  actual = subset_test_label_df[,1]
  predicted = xgb_predicted_df[,1]
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






