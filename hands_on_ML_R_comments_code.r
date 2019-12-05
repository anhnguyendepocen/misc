# Helper packages
library(dplyr)     # for data manipulation
library(ggplot2)   # for awesome graphics

# Modeling process packages
library(rsample)   # for resampling procedures
library(caret)     # for resampling and model training
library(h2o)       # for resampling and model training

# h2o set-up 
h2o.no_progress()  # turn off h2o progress bars
h2o.init()

# Ames housing data
ames <- AmesHousing::make_ames()
ames.h2o <- as.h2o(ames)

# Job attrition data
churn <- rsample::attrition %>% 
    mutate_if(is.ordered, .funs = factor, ordered = FALSE)
churn.h2o <- as.h2o(churn)

# Using base R
set.seed(123)  # for reproducibility
index_1 <- sample(1:nrow(ames), round(nrow(ames) * 0.7))
train_1 <- ames[index_1, ]
test_1  <- ames[-index_1, ]

# Using caret package
set.seed(123)  # for reproducibility
index_2 <- createDataPartition(ames$Sale_Price, p = 0.7, 
                               list = FALSE)
train_2 <- ames[index_2, ]
test_2  <- ames[-index_2, ]

# Using rsample package
set.seed(123)  # for reproducibility
split_1  <- initial_split(ames, prop = 0.7)
train_3  <- training(split_1)
test_3   <- testing(split_1)

# Using h2o package
split_2 <- h2o.splitFrame(ames.h2o, ratios = 0.7, 
                          seed = 123)
train_4 <- split_2[[1]]
test_4  <- split_2[[2]]


# Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# Specify resampling strategy
cv <- trainControl(
    method = "repeatedcv", 
    number = 10, 
    repeats = 5
)

# Create grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

# Tune a knn model using grid search
knn_fit <- train(
    Sale_Price ~ ., 
    data = ames_train, 
    method = "knn", 
    trControl = cv, 
    tuneGrid = hyper_grid,
    metric = "RMSE"
)


# Print and plot the CV results
knn_fit

ggplot(knn_fit)

####CHAPTER 3

# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics
library(visdat)   # for additional visualizations
library(tidyverse)

# Feature engineering packages
library(caret)    # for various ML tasks
library(recipes)  # for feature engineering tasks

###
# log transformation
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_log(all_outcomes())

ames_recipe

# Box Cox transform a value
y <- forecast::BoxCox(10, lambda)

# Inverse Box Cox function
inv_box_cox <- function(x, lambda) {
    # for Box-Cox, lambda = 0 --> log transform
    if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda) 
}

# Undo Box Cox-transformation
inv_box_cox(y, lambda)
## [1] 10
## attr(,"lambda")
## [1] 0.05259888

###
AmesHousing::ames_raw %>%
    is.na() %>%
    reshape2::melt() %>%
    ggplot(aes(Var2, Var1, fill=value)) + 
    geom_raster() + 
    coord_flip() +
    scale_y_continuous(NULL, expand = c(0, 0)) +
    scale_fill_grey(name = "", 
                    labels = c("Present", 
                               "Missing")) +
    xlab("Observation") +
    theme(axis.text.y  = element_text(size = 4))

# The vis_miss() function in R package visdat (Tierney 2019) also allows for easy visualization of missing data patterns (with sorting and clustering options). We illustrate this functionality below using the raw Ames housing data (Figure 3.4). The columns of the heat map represent the 82 variables of the raw data and the rows represent the observations. Missing values (i.e., NA) are indicated via a black cell. The variables and NA patterns have been clustered by rows (i.e., cluster = TRUE).

vis_miss(AmesHousing::ames_raw, cluster = TRUE)

ames_recipe %>%
    step_medianimpute(Gr_Liv_Area)

ames_recipe %>%
    step_knnimpute(all_predictors(), neighbors = 6)

ames_recipe %>%
    step_bagimpute(all_predictors())

###

caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>% 
    rownames_to_column() %>% 
    filter(nzv)

###

When normalizing many variables, it’s best to use the Box-Cox (when feature values are strictly positive) or Yeo-Johnson (when feature values are not strictly positive) procedures as these methods will identify if a transformation is required and what the optimal transformation 

# Normalize all numeric columns
recipe(Sale_Price ~ ., data = ames_train) %>%
    step_YeoJohnson(all_numeric())    

##standardize

ames_recipe %>%
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes())

##lumping
lumping <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_other(Neighborhood, threshold = 0.01, 
               other = "other") %>%
    step_other(Screen_Porch, threshold = 0.1, 
               other = ">0")

# Apply this blue print --> you will learn about this at 
# the end of the chapter
apply_2_training <- prep(lumping, training = ames_train) %>%
    bake(ames_train)

# Lump levels for two features
recipe(Sale_Price ~ ., data = ames_train) %>%
    step_dummy(all_nominal(), one_hot = TRUE)


#Label encoding is a pure numeric conversion of the levels of a categorical variable.
# Label encoded
recipe(Sale_Price ~ ., data = ames_train) %>%
    step_integer(MS_SubClass) %>%
    prep(ames_train) %>%
    bake(ames_train) %>%
    count(MS_SubClass)


##Target encoding runs the risk of data leakage since you are using the response variable to encode a feature. An alternative to this is to change the feature value to represent the proportion a particular level represents for a given feature. In this case, North_Ames would be changed to 0.153.

##pca
recipe(Sale_Price ~ ., data = ames_train) %>%
    step_center(all_numeric()) %>%
    step_scale(all_numeric()) %>%
    step_pca(all_numeric(), threshold = .95)


##While your project’s needs may vary, here is a suggested order of potential steps that should work for most problems:

# 1.Filter out zero or near-zero variance features.
# 2.Perform imputation if required.
# 3.Normalize to resolve numeric feature skewness.
# 4.Standardize (center and scale) numeric features.
# 5.Perform dimension reduction (e.g., PCA) on numeric features.
# 6.One-hot or dummy encode categorical features.
# will be.

##Data leakage is when information from outside the training data set is used to create the model. Data leakage often occurs during the data preprocessing period. To minimize this, feature engineering should be done in isolation of each resampling iteration. Recall that resampling allows us to estimate the generalizable prediction error. Therefore, we should apply our feature engineering blueprint to each resample independently

# 
# There are three main steps in creating and applying feature engineering with recipes:
#     
# 1.recipe: where you define your feature engineering steps to create your blueprint.
# 2. prep are: estimate feature engineering parameters based on training data.
# 3.bake: apply the blueprint to new data.

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_nzv(all_nominal())  %>%
    step_integer(matches("Qual|Cond|QC|Qu")) %>%
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes()) %>%
    step_pca(all_numeric(), -all_outcomes())

prepare <- prep(blueprint, training = ames_train)
prepare

baked_train <- bake(prepare, new_data = ames_train)
baked_test <- bake(prepare, new_data = ames_test)
baked_train

##Consequently, the goal is to develop our blueprint, then within each resample iteration we want to apply prep() and bake() to our resample training and validation data. Luckily, the caret package simplifies this process. We only need to specify the blueprint and caret will automatically prepare and bake within each resample. We illustrate with the ames housing example.

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_nzv(all_nominal()) %>%
    step_integer(matches("Qual|Cond|QC|Qu")) %>%
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

# Specify resampling plan
cv <- trainControl(
    method = "repeatedcv", 
    number = 10, 
    repeats = 5
)

# Construct grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

# Tune a knn model using grid search
knn_fit2 <- train(
    blueprint, 
    data = ames_train, 
    method = "knn", 
    trControl = cv, 
    tuneGrid = hyper_grid,
    metric = "RMSE"
)

knn_fit2



####Most statistical software, including R, will include estimated standard errors, t-statistics, etc. as part of its regression output. However, it is important to remember that such quantities depend on three major assumptions of the linear regresion model:
# Independent observations
# The random errors have mean zero, and constant variance
# The random errors are normally distributed
# If any or all of these assumptions are violated, then remdial measures need to be taken. For instance, weighted least squares (and other procedures) can be used when the constant variance assumption is violated. Transformations (of both the response and features) can also help to correct departures from these assumptions. The residuals are extremely useful in helping to identify how parametric models depart from such assumptions.

# perform 10-fold cross validation on a PLS model tuning the 
# number of principal components to use as predictors from 1-20
set.seed(123)
cv_model_pls <- train(
    Sale_Price ~ ., 
    data = ames_train, 
    method = "pls",
    trControl = trainControl(method = "cv", number = 10),
    preProcess = c("zv", "center", "scale"),
    tuneLength = 20
)

# model with lowest RMSE
cv_model_pls$bestTune
##   ncomp
## 3     3

# plot cross-validated RMSE
ggplot(cv_model_pls)



df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)

# Create training (70%) and test (30%) sets for the 
# rsample::attrition data.
set.seed(123)  # for reproducibility
churn_split <- initial_split(df, prop = .7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)
##Comparison models
set.seed(123)
cv_model1 <- train(
    Attrition ~ MonthlyIncome, 
    data = churn_train, 
    method = "glm",
    family = "binomial",
    trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model2 <- train(
    Attrition ~ MonthlyIncome + OverTime, 
    data = churn_train, 
    method = "glm",
    family = "binomial",
    trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model3 <- train(
    Attrition ~ ., 
    data = churn_train, 
    method = "glm",
    family = "binomial",
    trControl = trainControl(method = "cv", number = 10)
)

# extract out of sample performance measures
summary(
    resamples(
        list(
            model1 = cv_model1, 
            model2 = cv_model2, 
            model3 = cv_model3
        )
    )
)$statistics$Accuracy

library(ROCR)

# Compute predicted probabilities
m1_prob <- predict(cv_model1, churn_train, type = "prob")$Yes
m3_prob <- predict(cv_model3, churn_train, type = "prob")$Yes

# Compute AUC metrics for cv_model1 and cv_model3
perf1 <- prediction(m1_prob, churn_train$Attrition) %>%
    performance(measure = "tpr", x.measure = "fpr")
perf2 <- prediction(m3_prob, churn_train$Attrition) %>%
    performance(measure = "tpr", x.measure = "fpr")

# Plot ROC curves for cv_model1 and cv_model3
plot(perf1, col = "black", lty = 2)
plot(perf2, add = TRUE, col = "blue")
legend(0.8, 0.2, legend = c("cv_model1", "cv_model3"),
       col = c("black", "blue"), lty = 2:1, cex = 0.6)




df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)

# Create training (70%) and test (30%) sets for the
# rsample::attrition data. Use set.seed for reproducibility
set.seed(123)
churn_split <- initial_split(df, prop = .7, strata = "Attrition")
train <- training(churn_split)
test  <- testing(churn_split)

# train logistic regression model
set.seed(123)
glm_mod <- train(
    Attrition ~ ., 
    data = train, 
    method = "glm",
    family = "binomial",
    preProc = c("zv", "center", "scale"),
    trControl = trainControl(method = "cv", number = 10)
)

# train regularized logistic regression model
set.seed(123)
penalized_mod <- train(
    Attrition ~ ., 
    data = train, 
    method = "glmnet",
    family = "binomial",
    preProc = c("zv", "center", "scale"),
    trControl = trainControl(method = "cv", number = 10),
    tuneLength = 10
)

# extract out of sample performance measures
summary(resamples(list(
    logistic_model = glm_mod, 
    penalized_model = penalized_mod
)))$statistics$Accuracy

library(pdp)           # for partial dependence plots
library(vip)           # for variable importance plots

ggplot(penalized_mod)
varImp(penalized_mod)
vip(penalized_mod)

partial(penalized_mod, pred.var = "OverTime", plot = TRUE,
        plot.engine = "ggplot2")
partial(penalized_mod, pred.var = "TotalWorkingYears", plot = TRUE,
        plot.engine = "ggplot2")

###MARS model
set.seed(123)
ames <- AmesHousing::make_ames()

split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)
# create a tuning grid
hyper_grid <- expand.grid(
    degree = 1:3, 
    nprune = seq(2, 100, length.out = 10) %>% floor()
)

head(hyper_grid)

# Cross-validated model
set.seed(123)  # for reproducibility
cv_mars <- train(
    x = subset(ames_train, select = -Sale_Price),
    y = ames_train$Sale_Price,
    method = "earth",
    metric = "RMSE",
    trControl = trainControl(method = "cv", number = 10),
    tuneGrid = hyper_grid
)

# View results
cv_mars$bestTune
##    nprune degree
## 25     45      3
ggplot(cv_mars)

# variable importance plots
p1 <- vip(cv_mars, num_features = 40, bar = FALSE, value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 40, bar = FALSE, value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)

coef(cv_mars$finalModel)

p1 <- partial(cv_mars, pred.var = "Gr_Liv_Area", grid.resolution = 10) %>% autoplot()
p2 <- partial(cv_mars, pred.var = "Year_Built", grid.resolution = 10) %>% autoplot()
p3 <- partial(cv_mars, pred.var = c("Gr_Liv_Area", "Year_Built"), grid.resolution = 10) %>% 
    plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, screen = list(z = -20, x = -60))




set.seed(123)
index <- sample(nrow(mnist$train$images), size = 10000)
mnist_x <- mnist$train$images[index, ]
mnist_y <- factor(mnist$train$labels[index])


# The basic algorithm for a regression or classification random forest can be generalized as follows:
#     
#     1.  Given a training data set
# 2.  Select number of trees to build (n_trees)
# 3.  for i = 1 to n_trees do
# 4.  |  Generate a bootstrap sample of the original data
# 5.  |  Grow a regression/classification tree to the bootstrapped data
# 6.  |  for each split do
# 7.  |  | Select m_try variables at random from all p variables
# 8.  |  | Pick the best variable/split-point among the m_try
# 9.  |  | Split the node into two child nodes
# 10. |  end
# 11. | Use typical tree model stopping criteria to determine when a 
# | tree is complete (but do not prune)
# 12. end
# 13. Output ensemble of trees 

# 1. A good rule of thumb is to start with 10 times the number of features as illustrated
# 2.mtry With regression problems the default value is often  
# mtry=p/3 and for classification. However, when there are fewer relevant predictors (e.g., noisy data) a higher value of mtry tends to perform better because it makes it more likely to select those features with the strongest signal. When there are many relevant predictors, a lower mtry  might perform better.
# 3.Tree complexity (node size 1-10) Moreover, if computation time is a concern then you can often decrease run time substantially by increasing the node size and have only marginal impacts to your error estimate

# 11.4.4 Sampling scheme
# -Sample without replacement
# -Assess 3–4 values of sample sizes ranging from 25%–100% and if you have unbalanced categorical features try sampling without replacement.

# create hyperparameter grid
library(ranger)
n_features <- length(setdiff(names(ames_train), "Sale_Price"))


hyper_grid <- expand.grid(
    mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
    min.node.size = c(1, 3, 5, 10), 
    replace = c(TRUE, FALSE),                               
    sample.fraction = c(.5, .63, .8),                       
    rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
    # fit model for ith hyperparameter combination
    fit <- ranger(
        formula         = Sale_Price ~ ., 
        data            = ames_train, 
        num.trees       = n_features * 10,
        mtry            = hyper_grid$mtry[i],
        min.node.size   = hyper_grid$min.node.size[i],
        replace         = hyper_grid$replace[i],
        sample.fraction = hyper_grid$sample.fraction[i],
        verbose         = FALSE,
        seed            = 123,
        respect.unordered.factors = 'order',
    )
    # export OOB error 
    hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# assess top 10 models
hyper_grid %>%
    arrange(rmse) %>% 
    head(10)

# re-run model with impurity-based variable importance
rf_impurity <- ranger(
    formula = Sale_Price ~ ., 
    data = ames_train, 
    num.trees = 2000,
    mtry = 12,
    min.node.size = 1,
    sample.fraction = .80,
    replace = FALSE,
    importance = "impurity",
    respect.unordered.factors = "order",
    verbose = FALSE,
    seed  = 123
)

# re-run model with permutation-based variable importance
rf_permutation <- ranger(
    formula = Sale_Price ~ ., 
    data = ames_train, 
    num.trees = 2000,
    mtry = 12,
    min.node.size = 1,
    sample.fraction = .80,
    replace = FALSE,
    importance = "permutation",
    respect.unordered.factors = "order",
    verbose = FALSE,
    seed  = 123
)

p1 <- vip::vip(rf_impurity, num_features = 25, bar = FALSE)
p2 <- vip::vip(rf_permutation, num_features = 25, bar = FALSE)

gridExtra::grid.arrange(p1, p2, nrow = 1)


####
Sequential training with respect to errors: Boosted trees are grown sequentially; each tree is grown using information from previously grown trees to improve performance. This is illustrated in the following algorithm for boosting regression trees. By fitting each tree in the sequence to the previous tree’s residuals, we’re allowing each new tree in the sequence to focus on the previous tree’s mistakes:
    
# 1. Fit a decision tree to the data:  
# 2. We then fit the next decision tree to the residuals of the previous: 
# 3. Add this new tree to our algorithm:  
# 4. Fit the next decision tree to the residuals of  
# 5. Add this new tree to our algorithm:  
# 6. Continue this process until some mechanism (i.e. cross validation) tells us to stop.
#The final model here is a stagewise additive model of b individual trees:
    
    
#Many algorithms in regression, including decision trees, focus on minimizing some function of the residuals; most typically the SSE loss function, or equivalently, the MSE or RMSE (this is accomplished through simple calculus and is the approach taken with least squares).
# Gradient boosting is considered a gradient descent algorithm. Gradient descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems.

# Gradient descent can be performed on any loss function that is differentiable. Consequently, this allows GBMs to optimize different loss functions as desired (see J. Friedman, Hastie, and Tibshirani (2001), p. 360 for common loss functions). An important parameter in gradient descent is the size of the steps which is controlled by the learning rate. If the learning rate is too small, then the algorithm will take many iterations (steps) to find the minimum. On the other hand, if the learning rate is too high, you might jump across the minimum and end up further away than when you started.

##Number of trees:depending on the values of the other hyperparameters, GBMs often require many trees (it is not uncommon to have many thousands of trees) but since they can easily overfit we must find the optimal number of trees that minimize the loss function of interest with cross validation.

##Learning rate:  Smaller values make the model robust to the specific characteristics of each individual tree, thus allowing it to generalize well. Smaller values also make it easier to stop prior to overfitting; however, they increase the risk of not reaching the optimum with a fixed number of trees and are more computationally demanding. This hyperparameter is also called shrinkage. Generally, the smaller this value, the more accurate the model can be but also will require more trees in the sequence.

##Tree depth: Controls the depth of the individual trees. Typical values range from a depth of 3–8 but it is not uncommon to see a tree depth of 1 (J. Friedman, Hastie, and Tibshirani 2001). Smaller depth trees such as decision stumps are computationally efficient (but require more trees); however, higher depth trees allow the algorithm to capture unique interactions but also increase the risk of over-fitting.
##Minimum number of observations in terminal nodes: Also, controls the complexity of each tree. Since we tend to use shorter trees this rarely has a large impact on performance. Typical values range from 5–15 where higher values help prevent a model from learning relationships which might be highly specific to the particular sample selected for a tree (overfitting) but smaller values can help with imbalanced target classes in classification problems.

# General tuning strategy
# Choose a relatively high learning rate. Generally the default value of 0.1 works but somewhere between 0.05–0.2 should work across a wide range of problems.
# Determine the optimum number of trees for this learning rate.
# Fix tree hyperparameters and tune learning rate and assess speed vs. performance.
# Tune tree-specific parameters for decided learning rate.
# Once tree-specific parameters have been found, lower the learning rate to assess for any improvements in accuracy.
# Use final hyperparameter settings and increase CV procedures to get more robust estimates. Often, the above steps are performed with a simple validation procedure or 5-fold CV due to computational constraints. If you used k-fold CV throughout steps 1–5 then this step is not necessary.

##An important insight made by Breiman (Breiman (1996a); Breiman (2001)) in developing his bagging and random forest algorithms was that training the algorithm on a random subsample of the training data set offered additional reduction in tree correlation and, therefore, improvement in prediction accuracy. 
hyper_grid <- list(
    sample_rate = c(0.5, 0.75, 1),              # row subsampling
    col_sample_rate = c(0.5, 0.75, 1),          # col subsampling for each split
    col_sample_rate_per_tree = c(0.5, 0.75, 1)  # col subsampling for each tree
)

##xgboost provides multiple regularization parameters to help reduce model complexity and guard against overfitting. The first, gamma, is a pseudo-regularization hyperparameter known as a Lagrangian multiplier and controls the complexity of a given tree. gamma specifies a minimum loss reduction required to make a further partition on a leaf node of the tree. When gamma is specified, xgboost will grow the tree to the max depth specified but then prune the tree to find and remove splits that do not meet the specified gamma. Two more traditional regularization parameters include alpha and lambda. alpha provides an L1 regularization (reference Section 6.2.2) and lambda provides an L2 regularization (reference Section 6.2.1).

# The general tuning strategy for exploring xgboost hyperparameters builds onto the basic and stochastic GBM tuning strategies:
#     
# 1. Crank up the number of trees and tune learning rate with early stopping
# 2. Tune tree-specific hyperparameters
# 3. Explore stochastic GBM attributes
# 4. If substantial overfitting occurs (e.g., large differences between train and CV error) explore regularization hyperparameters
# 5. If you find hyperparameter values that are substantially different from default settings, be sure to retune the learning rate
# 6.Obtain final “optimal” model

library(xgboost)
library(recipes)
xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_integer(all_nominal()) %>%
    prep(training = ames_train, retain = TRUE) %>%
    juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Sale_Price")])
Y <- xgb_prep$Sale_Price
# hyperparameter grid
hyper_grid <- expand.grid(
    eta = 0.01,
    max_depth = 3, 
    min_child_weight = 3,
    subsample = 0.5, 
    colsample_bytree = 0.5,
    gamma = c(0, 1, 10, 100, 1000),
    lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
    alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
    rmse = 0,          # a place to dump RMSE results
    trees = 0          # a place to dump required number of trees
)

# grid search
for(i in seq_len(nrow(hyper_grid))) {
    set.seed(123)
    m <- xgb.cv(
        data = X,
        label = Y,
        nrounds = 4000,
        objective = "reg:linear",
        early_stopping_rounds = 50, 
        nfold = 10,
        verbose = 0,
        params = list( 
            eta = hyper_grid$eta[i], 
            max_depth = hyper_grid$max_depth[i],
            min_child_weight = hyper_grid$min_child_weight[i],
            subsample = hyper_grid$subsample[i],
            colsample_bytree = hyper_grid$colsample_bytree[i],
            gamma = hyper_grid$gamma[i], 
            lambda = hyper_grid$lambda[i], 
            alpha = hyper_grid$alpha[i]
        ) 
    )
    hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
    hyper_grid$trees[i] <- m$best_iteration
}

# results
hyper_grid %>%
    filter(rmse > 0) %>%
    arrange(rmse) %>%
    glimpse()


# optimal parameter list
params <- list(
    eta = 0.01,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.5,
    colsample_bytree = 0.5
)

# train final model
xgb.fit.final <- xgboost(
    params = params,
    data = X,
    label = Y,
    nrounds = 3944,
    objective = "reg:linear",
    verbose = 0
)

# xgboost actually provides three built-in measures for feature importance:
#     
#     Gain: This is equivalent to the impurity measure in random forests (reference Section 11.6) and is the most common model-centric metric to use.
# Coverage: The Coverage metric quantifies the relative number of observations influenced by this feature.
# Frequency: The percentage representing the relative number of times a particular feature occurs in the trees of the model.

# 
# SVMs have a number of advantages compared to other ML algorithms described in this book. First off, they attempt to directly maximize generalizability (i.e., accuracy). Since SVMs are essentially just convex optimization problems, we’re always guaranteed to find a global optimum (as opposed to potentially getting stuck in local optima as with DNNs). By softening the margin using a budget (or cost) parameter (  C), SVMs are relatively robust to outliers. And finally, using kernel functions, SVMs are flexible enough to adapt to complex nonlinear decision boundaries (i.e., they can flexibly model nonlinear relationships). However, SVMs do carry a few disadvantages as well. For starters, they can be slow to train on tall data (i.e.,  n>p)
# ). This is because SVMs essentially have to estimate at least one parameter for each row in the training data! Secondly, SVMs only produce predicted class labels; obtaining predicted class probabilities requires additional adjustments and computations not covered in this chapter. Lastly, special procedures (e.g., OVA and OVO) have to be used to handle multinomial classification problems with SVMs.

# Control params for SVM
ctrl <- trainControl(
    method = "cv", 
    number = 10, 
    classProbs = TRUE,                 
    summaryFunction = twoClassSummary  # also needed for AUC/ROC
)

# Tune an SVM
set.seed(5628)  # for reproducibility
churn_svm_auc <- train(
    Attrition ~ ., 
    data = churn_train,
    method = "svmRadial",               
    preProcess = c("center", "scale"),  
    metric = "ROC",  # area under ROC curve (AUC)       
    trControl = ctrl,
    tuneLength = 10
)

# Print results
churn_svm_auc$results

prob_yes <- function(object, newdata) {
    predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

# Variable importance plot
set.seed(2827)  # for reproducibility
vip(churn_svm_auc, method = "permute", nsim = 5, train = churn_train, 
    target = "Attrition", metric = "auc", reference_class = "Yes", 
    pred_wrapper = prob_yes)

features <- c("OverTime", "WorkLifeBalance", 
              "JobSatisfaction", "JobRole")
pdps <- lapply(features, function(x) {
    partial(churn_svm_auc, pred.var = x, which.class = 2,  
            prob = TRUE, plot = TRUE, plot.engine = "ggplot2") +
        coord_flip()
})
grid.arrange(grobs = pdps,  ncol = 2)



# Stacking (sometimes called “stacked generalization”) involves training a new learning algorithm to combine the predictions of several base learners. First, the base learners are trained using the available training data, then a combiner or meta algorithm, called the super learner, is trained to make a final prediction based on the predictions of the base learners. Such stacked ensembles tend to outperform any of the individual base learners (e.g., a single RF or GBM) and have been shown to represent an asymptotically optimal system for learning (Laan, Polley, and Hubbard 2003).



# Load and split the Ames housing data
ames <- AmesHousing::make_ames()
set.seed(123)  # for reproducibility
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

# Make sure we have consistent categorical levels
# blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
#     step_other(all_nominal(), threshold = 0.005)

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_nzv(all_nominal()) %>%
    step_integer(matches("Qual|Cond|QC|Qu")) %>%
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes()) %>%
    step_other(all_nominal(), threshold = 0.005)

# Create training & test sets for h2o
train_h2o <- prep(blueprint, training = ames_train, retain = TRUE) %>%
    juice() %>%
    as.h2o()
test_h2o <- prep(blueprint, training = ames_train) %>%
    bake(new_data = ames_test) %>%
    as.h2o()

# Get response and feature names
Y <- "Sale_Price"
X <- setdiff(names(train_h2o), Y)
# Use AutoML to find a list of candidate models (i.e., leaderboard)
auto_ml <- h2o.automl(
    x = X, y = Y, training_frame = train_h2o, nfolds = 5, 
    max_runtime_secs = 60, max_models = 50,
    keep_cross_validation_predictions = TRUE, sort_metric = "RMSE", seed = 123,
    stopping_rounds = 50, stopping_metric = "RMSE", stopping_tolerance = 0
)

# Assess the leader board; the following truncates the results to show the top 
# and bottom 15 models. You can get the top model with auto_ml@leader
auto_ml@leaderboard %>% 
    as.data.frame() %>%
    dplyr::select(model_id, rmse) %>%
    dplyr::slice(1:25)

