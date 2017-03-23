packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

library(syuzhet)
library(DT)

data <- fromJSON("../input/train.json")
vars <- setdiff(names(data), c("photos", "features"))
train_df <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)

data <- fromJSON("../input/test.json")
vars <- setdiff(names(data), c("photos", "features"))
test_df <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)

sentiment_train <- get_nrc_sentiment(train_df$description)
joint_train <- cbind(train_df$listing_id,sentiment_train)
write(toJSON(joint_train),"../input/train_description_sentiment.json")

sentiment_test <- get_nrc_sentiment(test_df$description)
joint_test <- cbind(test_df$listing_id,sentiment_test)
write(toJSON(joint_test),"../input/test_description_sentiment.json")