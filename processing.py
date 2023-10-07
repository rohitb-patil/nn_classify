import pandas as pd
import torch
import pandas as pd
import torch
import numpy as np
###################### Additional Imports ####################
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

def data_preprocessing(file_path):
    # 1. Load the CSV dataset
    df = pd.read_csv(file_path)

    # # 2. Remove unwanted features (If there are any specific columns you want to remove, add them in the list below)
    # unwanted_columns = []  # Add column names here if needed
    # df.drop(columns=unwanted_columns, inplace=True)

    # 3. Handle missing values
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical (textual) column
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)
        else:  # Numerical column
            mean_val = df[column].mean()
            df[column].fillna(mean_val, inplace=True)

    # Convert JoiningYear to Years at Company (if 'JoiningYear' is in the dataset)
    if 'JoiningYear' in df.columns:
        current_year = pd.Timestamp.now().year
        df['YearsAtCompany'] = current_year - df['JoiningYear']
        df.drop('JoiningYear', axis=1, inplace=True)

    # 4. Encode textual features numerically
    def encode_category(column):
        unique_vals = column.unique()
        val_to_int = {val: i for i, val in enumerate(unique_vals)}
        return column.map(val_to_int), val_to_int

    encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':  # It's a categorical (textual) column
            df[column], encoders[column] = encode_category(df[column])

    return df, encoders

def identify_features_and_targets(encoded_dataframe):
	  # The target label is the Salary column.
    target_column = 'LeaveOrNot'

  # The features are all the other columns in the dataframe.
    feature_columns = list(encoded_dataframe.columns.drop(target_column))
    # Extract the selected features and target label from the DataFrame
    features = encoded_dataframe[feature_columns]
    target = encoded_dataframe[target_column]

    # Create a list containing the features and the target
    features_and_targets = [features, target]
    return features_and_targets

def load_as_tensors(features_and_targets):
	# Extract features and target from the input list
    features, target = features_and_targets
	 # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.75, random_state=42)

    # Convert the training and validation data to PyTorch tensors.
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
	
    # Create an iterable dataset object for the training data.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # Create a DataLoader object for the training data.
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
	
    # Create the list of tensors and iterable data loaders
    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_dataloader, val_dataloader]

    return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
    def __init__(self):
        super(Salary_Predictor, self).__init__()	
        # Define the input layer.
        self.input_layer = nn.Linear(8, 32)
        # Define the hidden layer.
        self.hidden_layer = nn.Linear(32, 64)
        self.hidden_layer1 = nn.Linear(64, 32)
        self.hidden_layer2= nn.Linear(32, 8)
        # Define the output layer.
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        # Pass the input data through the input layer.
        x = F.relu(self.input_layer(x))
        # Apply the ReLU activation function.
        x = F.relu(self.hidden_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
       #Getting the output layer
        x = self.output_layer(x)
        # Return the predicted salary.
        predicted_output = x

        return predicted_output
	
def model_loss_function():
        loss_function = nn.MSELoss()
        return loss_function

def model_optimizer(model):
	optimizer = optim.Adam(model.parameters())
	return optimizer

def model_number_of_epochs():
	# Define the number of epochs.
    number_of_epochs = 10

    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	# Unpack tensors and data loader from the provided list
    X_train, X_test, y_train, y_test, train_dataloader ,val_dataloader = tensors_and_iterable_training_data

    # Train the model for the specified number of epochs.
    for epoch in range(number_of_epochs):
        # Iterate over the training data in batches.
        for batch_idx, (x, y) in enumerate(train_dataloader):
            # Forward pass.
            predictions = model.forward(x)  # Use .forward() to get predictions.

            # Calculate the loss.
            loss = loss_function(predictions, y)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the training progress.
            if batch_idx % 100 == 0:
                print('Training epoch: {} | Batch index: {} | Loss: {}'.format(epoch, batch_idx, loss.item()))

    # Return the trained model with the name 'trained_model'.
    trained_model = model  # Assign the model to the name 'trained_model' before returning.
    return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
	# Unpack tensors and data loader from the provided list
    _, X_val_tensor, _, y_val_tensor, _, val_data_loader = tensors_and_iterable_training_data

     # Set the model to evaluation mode
    trained_model.eval()

    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculation during validation
    with torch.no_grad():
        for batch_X, batch_y in val_data_loader:
            # Make predictions using the trained model
            outputs = trained_model(batch_X)

            # Get the predicted class labels
            _, predicted = torch.max(outputs, 1)

            # Count the number of correct predictions in this batch
            correct_predictions += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

    # Calculate the accuracy on the validation dataset
    model_accuracy = (correct_predictions / total_samples) * 100.0

    # Print the validation accuracy
    #print(f'Validation Accuracy: {model_accuracy:.2f}%')
    
    return model_accuracy


# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''""""""""""""""""""""""""""""""""""""""
# Example usage:
task_1a_dataframe = 'E:/Task_1A/task_1a_dataset.csv'
# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe

	# data preprocessing and obtaining encoded data
encoded_dataframe , encoders = data_preprocessing(task_1a_dataframe)
print("data ==== ",encoded_dataframe)
	# selecting required features and targets
features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
loss_function = model_loss_function()
optimizer = model_optimizer(model)
number_of_epochs = model_number_of_epochs()

	# training the model
trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
print(f"Accuracy on the test set = {model_accuracy}")