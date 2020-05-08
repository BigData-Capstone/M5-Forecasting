#importing relevant libraries
library(tidyverse)
library(data.table)
library(RcppRoll)
library(dplyr)

train <- fread("sales_train_validation.csv", stringsAsFactors = FALSE)
free <- function() invisible(gc())
train <- train %>% 
  #eliminate valitrainion to save memory
  mutate(id = gsub("_valitrainion", "", id)) %>%
  select(-item_id) %>% 
  select(-dept_id) %>%
  select(-cat_id) %>%
  select(-store_id) %>%
  select(-state_id)

View(train)
free()
#transpose the data
train = t(train)
free()
#write first row as column header



View(train)
