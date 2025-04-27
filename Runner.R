### (MUST BE CHECKED) Local setup
source_path <- "Sourced.R"
library(tfruns)

### (MUST BE CHECKED) Define values for hyperparameters to be tuned
hyper <- list(
  batch_size = c(16, 32, 64),
  epochs = c(10, 20, 30),
  dropout_rate = c(0.2, 0.3, 0.5),
  learning_rate = c(1e-3, 5e-4, 1e-4),
  reduce_lr_factor = c(0.5, 0.3),
  reduce_lr_patience = c(2, 3, 5)
)

# How many options for each parameter
lengths <- sapply(hyper, length)

# Total combinations = product of lengths
total_combinations <- prod(lengths)

cat("Total combinations:", total_combinations, "\n")

### (MUST BE CHECKED) Number of runs I want to do
wanted_runs <- 30

# Compute sample fraction
sample_fraction <- wanted_runs / total_combinations

cat("Sample fraction to use:", sample_fraction, "\n")

# Only used for testing purposes (check everything works fine)
set.seed(123)
chosen_hyper <- list(
  batch_size = sample(hyper$batch_size, 1),
  epochs = sample(hyper$epochs, 1),
  dropout_rate = sample(hyper$dropout_rate, 1),
  learning_rate = sample(hyper$learning_rate, 1),
  reduce_lr_factor = sample(hyper$reduce_lr_factor, 1),
  reduce_lr_patience = sample(hyper$reduce_lr_patience, 1)
)

# Arguments explanation:
# 1) runs_dir: Directory where runs are gonna be saved
# 2) sample: If too much hyperparameters, the combination of all of'em would be a big grid
# So just check a sample of them
# 3) confirm: To avoid that the function asks me for my permission before running
runs <- tuning_run(
  source_path, 
  runs_dir = "_tuning", 
  flags = hyper, 
  sample = sample_fraction,
  confirm = FALSE
)

# View summary of runs results (we get a df)
ls_runs_df <- ls_runs(runs_dir = "_tuning")
ls_runs_df