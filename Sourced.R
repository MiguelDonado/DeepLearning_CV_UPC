# Local specific setup
# setwd("/home/miguelUPC/Q2/DNN/Trabajos/4.DeepLearning_CV/")
library(reticulate)
use_virtualenv("/home/miguel/.venvs/deep_neural_network3.9")
# Check Python is being properly executed in R
py_config()
data_path <- "/seg_pred/" 

# ML/AI specific setup
library(tensorflow)
library(keras3)
library(tfdatasets)    # Loaded for mapping label = image step

# Callbacks and tfruns flags
FLAGS <- flags(
  # Training paremeters
  flag_integer("batch_size", default = 32),
  flag_integer("epochs", default = 3),
  
  # Learning rate
  flag_numeric("learning_rate", default = 0.0001),
  
  # Dropout rate
  flag_numeric("dropout_rate", default = 0.2),
  
  # Callbacks: ReduceLROnPlateau factor
  flag_numeric("reduce_lr_factor", default = 0.5), 
  
  # Callbacks: ReduceLROnPlateau patience
  flag_integer("reduce_lr_patience", default = 2)
)

callbacks_list <- list(
  callback_early_stopping(
    monitor = "val_loss", 
    patience = 5,             
  ),
  callback_model_checkpoint(
    filepath = "best_model.keras",
    monitor = "val_loss",
    save_best_only = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = FLAGS$reduce_lr_factor,
    patience = FLAGS$reduce_lr_patience,
  )
)

### Exercise 1

# Input image dimensions
img_rows <- 150
img_cols <- 150
input_dim <- c(150,150,3)
seed_train_validation = 1
shuffle_value = FALSE
validation_split = 0.3

## Load images. Explanation: https://stackoverflow.com/questions/66036271/splitting-a-tensorflow-dataset-into-training-test-and-validation-sets-from-ker
train_data <- image_dataset_from_directory(
  directory = train_path,
  image_size = c(img_rows,img_cols),
  validation_split = validation_split,
  subset = "training",
  seed = seed_train_validation,
  shuffle = shuffle_value,
  batch_size = FLAGS$batch_size,
  labels = NULL                                 # autoencoder (reconstruct input)
)
val_data <- image_dataset_from_directory(
  directory = train_path,
  image_size = c(img_rows,img_cols),
  validation_split = validation_split,
  subset = "validation",
  seed = seed_train_validation,
  shuffle = shuffle_value,
  batch_size = FLAGS$batch_size,
  labels = NULL                                 # autoencoder (reconstruct input)
)

# layer_rescaling(scale = 1/255) inside the model it doesnt work because only normalize inputs but not labels
# Normalize and prepare dataset (input = label)
train_data <- train_data %>% dataset_map(function(x) list(x / 255, x / 255)) %>% dataset_prefetch(buffer_size = tf$data$AUTOTUNE)
val_data <- val_data %>% dataset_map(function(x) list(x / 255, x / 255)) %>% dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

# Predicts expects only inputs, not pairs (input,label)
train_inputs <- train_data %>% dataset_map(function(x,y) x) 
val_inputs <- val_data %>% dataset_map(function(x,y) x) 

## Check properly loaded and save image for later prediction with model
# as_iterator: Converts train_data (special object in Tensorflow a tf.data.Dataset) on an iterator
# iter_next: Function to iterate over an iterator

batch <- as_iterator(train_data) %>% iter_next()
str(batch)

## Plot an image
original_images <- batch[[1]] 
original_image <- original_images[1,,,]
original_image_batch <- tf$expand_dims(original_image, axis = as.integer(0))
img_array <- as.array(original_image)  # Normalize to avoid function error
plot(as.raster(img_array))

########## Autoencoder
#### Convolutional encoder
input_img <- layer_input(shape = input_dim)
output_enc <- input_img %>% 
  layer_conv_2d(filters = 16, kernel_size = 3, activation = "relu", padding = "same") %>% 
  layer_droput(rate = FLAGS$dropout_rate) %>% 
  layer_max_pooling_2d(pool_size = 2, padding = "same") %>% 
  
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu", padding = "same") %>% 
  layer_dropout(rate = FLAGS$dropout_rate) %>% 
  layer_max_pooling_2d(pool_size = 2, padding = "same") %>% 

  layer_conv_2d(filters = 45, kernel_size = 3, activation = "relu", padding = "same") %>% 
  layer_dropout(rate = FLAGS$droput_rate) %>% 
  layer_max_pooling_2d(pool_size = 2, padding = "same") 

model_enc <- keras_model(inputs = input_img, outputs = output_enc)
summary(model_enc)

paste0('Flatten layer: ', 19 * 19 * 45, " units")

#### Convolutional decoder
encoded_input <- layer_input(shape = c(19, 19, 45)) # flatten output
output_dec <- encoded_input %>% 
  layer_conv_2d(filters = 45, kernel_size = 3, activation = "relu", padding = "same") %>% 
  
  layer_upsampling_2d(size = 2) %>% 
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu", padding = "same") %>% 
  
  layer_upsampling_2d(size = 2) %>%  
  layer_conv_2d(filters = 16, kernel_size = 3, activation = "relu", padding = "same") %>% 
  
  layer_upsampling_2d(size = 2) %>%  
  # On this last layer I use to sigmoid to force that values are between 0-1 range. For more info https://stackoverflow.com/questions/65307833/why-is-the-decoder-in-an-autoencoder-uses-a-sigmoid-on-the-last-layer
  layer_conv_2d(filters = 3, kernel_size = 3, activation = "sigmoid")

model_dec <- keras_model(inputs = encoded_input, outputs = output_dec)
summary(model_dec)

# Check input dimension == output dimension
input_dim <- unlist(model_enc$input_shape)
output_dim <- unlist(model_dec$output_shape)

if (identical(input_dim, output_dim)) {
  cat("Same dimensions:", input_dim, "\n")
} else {
  cat("Different dimensions:\nInput:", input_dim, "\nOutput:", output_dim, "\n")
}


#### Autoencoder

model <- keras_model_sequential()
model %>%  model_enc %>%  model_dec

summary(model)

########## Training
model %>% compile(
  optimizer = optimizer_adam(learning_rate = FLAGS$learning_rate),
  loss = "mean_squared_error",
  metrics = c("mean_squared_error")
)

history <- model %>% fit(
  x = train_data, # Autoencoder
  epochs = FLAGS$epochs,
  validation_data = val_data,       # Validation_data instead of validation_split
  callbacks = callbacks_list
)