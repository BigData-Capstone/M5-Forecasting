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


###helper functions
#define function for clearing memory
free <- function() invisible(gc())

#define mode function to find most frequent value
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#read in the data 
sales_train_validation.csv <- fread("sales_train_validation.csv", stringsAsFactors = TRUE)
calendar.csv <- fread("calendar.csv", stringsAsFactors = TRUE)
sell_prices.csv <- fread("sell_prices.csv", stringsAsFactors = TRUE)

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

#create features
dataset <- dataset %>%
  demand_features() 

#clear memory and remove csv's
rm(sales_train_validation.csv)
rm(calendar.csv)
rm(sell_prices.csv)
free()

#filter only the items of store CA_3
dataset = filter(dataset, store_id == "CA_3")

#drop un-nessecary colums store_id & state id & id
dataset = select(dataset, -store_id)
dataset = select(dataset, -state_id)

#clear memory again
free()

#create training data for simple_dataset -> Last 28 days should be predicted
index = tail(1:nrow(simple_dataset),28)
train_simple = simple_dataset[-index,]
test_simple = simple_dataset[index,]
View(test_simple)

#create training for the more complex dataset
#create a list of all the items
item_id <- data.frame(dataset$item_id)
item_id <- unique(item_id)

#Encode category and departement id as dummy variables
dataset$cat_id = one_hot(as.data.table(dataset$cat_id))
dataset$dept_id = one_hot(as.data.table(dataset$dept_id))

#split the training data
train_dataset = filter(dataset, d <= test_index)
test_dataset = filter(dataset, d > test_index)

#Assign label
train_label <- train_dataset$demand
test_label <- test_dataset$demand


#remove label from dataset
#train_dataset = select(train_dataset, -demand)
#test_dataset = select(test_dataset, -demand)

#gib mir nur ein produkt
View(train_dataset)
subset = filter(train_dataset, item_id == "HOBBIES_1_001")
View(subset)

#preparation for looping through the data
iterations = ncol(train_simple)
variables = 3
results_matrix <- matrix(ncol=variables, nrow=iterations)
computing_start_time <- Sys.time()

iterations = 3
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
  
  #loop to calculate rmse for every day
  for(x in 1:28){
    actual = new_test[x,1]
    predicted = predicted_df[x,1]
    performance = RMSE(actual, predicted)
    #write results in performance collector
    performance_collector[x,1] = performance
  }
  #write results into results matrix
  results_matrix[i,1] = mean(performance_collector)
  
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
  
  #loop to calculate rmse for every day
  for(x in 1:28){
    actual = new_test[x,1]
    snaive_predicted = snaive_predicted_df[x,1]
    performance = RMSE(actual, snaive_predicted)
    #write results in performance collector
    performance_collector[x,1] = performance
  }
  #write results into results matrix
  results_matrix[i,2] = mean(performance_collector)

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
    for(x in 1:28){
      actual = new_test[x,1]
      autoarima_predicted = autoarima_predicted_df[x,1]
      performance = RMSE(actual, autoarima_predicted)
      #write results in performance collector
      performance_collector[x,1] = performance
    }
    #write results into results matrix
    results_matrix[i,3] = mean(performance_collector)
  }, silent = FALSE)
  
  #########################################################################################
  ### Data Preparation for ML approaches
  #########################################################################################
  #select the individual product
  
  
  #########################################################################################
  ### Random Forest Forecast
  #########################################################################################
  
  #########################################################################################
  ### ML Approach Forecast
  #########################################################################################
  
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













