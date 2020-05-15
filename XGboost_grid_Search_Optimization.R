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
# View(dataset)


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
test_label = test_dataset$demand       # for grid search

#remove label from dataset
xg_train_dataset = select(train_dataset, -demand)
xg_test_dataset = select(test_dataset, -demand)       # for grid search

#convert datasets to matrix
x_train = as.matrix(xg_train_dataset)
x_test = as.matrix(xg_test_dataset)       # for grid search

#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = x_train, label = train_label)
testDMatrix <- xgb.DMatrix(data = x_test, label = test_label)       # for grid search

#set the parameter
params <- list(booster = "gbtree",
               objective = "reg:linear",
               eval_metric = "rmse",
               eta = 0.2,
               max_depth = 6,
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

# View feature immportance from the learnt model @moritz: https://cran.r-project.org/web/packages/xgboost/vignettes/xgboostPresentation.html
# importance_matrix <- xgb.importance(model = model_xgb)
# print(importance_matrix)
# xgb.plot.importance(importance_matrix = importance_matrix)

##### Custom Grid search https://datascienceplus.com/extreme-gradient-boosting-with-r/ 

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,                 #nfolds
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100),  # Parameter explanation: slides around 62 https://www.slideshare.net/ShangxuanZhang/kaggle-winning-solution-xgboost-algorithm-let-us-learn-from-its-author
                       max_depth = c(3, 4, 5, 6, 8, 10, 12, 15),           #default: 6
                       min_child_weight = c(5, 7, 9, 10),    #default: 1
                       colsample_bytree = seq(0.3, 0.7, by = 0.1),                     #default: 1          
                       ## default
                       eta = seq(0.05, 0.3, by = 0.05),
                       gamma = seq(0.0, 0.4, by = 0.1),
                       subsample = seq(0.7, 1, by = 0.1))


set.seed(123) 
xgb_model = train(
                  trainDMatrix,
                  train_label,  
                  trControl = xgb_trcontrol,
                  tuneGrid = xgbGrid,
                  method = "xgbTree",
                  matric = "RMSE",                       #vllt nicht nötig
                  maximize = FALSE,
                  trace = TRUE)

xgb_model$bestTune

predicted = predict(xgb_model, testDMatrix)
residuals = test_label - predicted
RMSE = sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE,3),'\n')


options(repr.plot.width=8, repr.plot.height=4)
my_data = as.data.frame(cbind(predicted = predicted,
                              observed = test_label))
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm)+ ggtitle('Linear Regression ') + ggtitle("Extreme Gradient Boosting: Prediction vs Test Data") +
  xlab("Predecited Power Output ") + ylab("Observed Power Output") + 
  theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
        axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))











