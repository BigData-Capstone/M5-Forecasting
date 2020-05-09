library(data.table)
library(ggplot2)

free <- function() invisible(gc())
create_dt <- function(is_train = TRUE, nrows = Inf) {
  
  # create train set
  dt <- fread("sales_train_validation.csv", nrows = nrows)
  cols <- dt[, names(.SD), .SDcols = patterns("^d_")]
  dt[, (cols) := transpose(lapply(transpose(.SD),
                                  function(x) {
                                    i <- min(which(x > 0))
                                    x[1:i-1] <- NA
                                    x})), .SDcols = cols]
  free()
  
  
  #omit missing values
  dt <- na.omit(melt(dt,
                     measure.vars = patterns("^d_"),
                     variable.name = "d",
                     value.name = "sales"))
  free()
  #read the calendar
  cal <- fread("calendar.csv")
  free()
  dt <- dt[cal, `:=`(date = as.IDate(i.date, format="%Y-%m-%d"), # merge tables by reference
                     wm_yr_wk = i.wm_yr_wk,
                     event_name_1 = i.event_name_1,
                     snap_CA = i.snap_CA,
                     snap_TX = i.snap_TX,
                     snap_WI = i.snap_WI), on = "d"]
  free()
  #read the prices
  prices <- fread("sell_prices.csv")
  dt[prices, sell_price := i.sell_price, on = c("store_id", "item_id", "wm_yr_wk")] # merge again
}

tr <- create_dt()
free()

#visualize
tr[, .(sales = unlist(lapply(.SD, sum))), by = "date", .SDcols = "sales"
   ][, ggplot(.SD, aes(x = date, y = sales)) +
       geom_line(size = 0.3, color = "steelblue", alpha = 0.8) + 
       geom_smooth(method='lm', formula= y~x, se = FALSE, linetype = 2, size = 0.5, color = "gray20") + 
       labs(x = "", y = "total sales") +
       theme_minimal() +
       theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position="none") +
       scale_x_date(labels=scales::date_format ("%b %y"), breaks=scales::date_breaks("3 months"))]