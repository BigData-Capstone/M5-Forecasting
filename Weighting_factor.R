#create weighting factor
dataset <- dataset %>%
  group_by(store_id) %>%
  mutate(
    total_sales_per_store = sum(demand)) %>%
  ungroup() %>%
  group_by(id) %>%
  mutate(
    total_sales_per_item = sum(demand),
    weighted_sales_percentage = total_sales_per_item/total_sales_per_store
  )%>%
  ungroup()