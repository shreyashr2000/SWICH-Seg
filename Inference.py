def main():
    # Initialize the UNet model
    model = UNet(in_channels=1, out_channels=2, init_features=16)
    
    # Load the state dictionary of the model from the saved file
    state_dict = torch.load(model_path)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    return model

  resnet_model = ResNet101()  # Initialize ResNet model

  resnet_model_from_gru = 'best_resnet_gru_model.pth'
# Create an instance of the LSTM-based classifier
    # 2. Train ResNet-GRU model
    best_resnet_model=remove_last_fc_layer(best_resnet_model)   #Remove fc layer from best_resnet_model
    resnet_gru_model = GRUClassifier(best_resnet_model,hidden_dim, output_dim) 
  best_resnet_fc_model= 'best_resnet_fc_model.pth'
  best_unet_model_path = 'best_unet_model.pth'  # Path to the saved model file
    # Initialize the UNet model
  model = UNet(in_channels=1, out_channels=2, init_features=16)
    # Load the state dictionary of the model from the saved file
  state_dict = torch.load(model_path)
