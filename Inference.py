def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    epochs = 100
    # Load INSTANCE dataset
    test_data,test_mask = data_load('','',large=False,train=False)
    test_dataset = test_lowdata_numpy_dataset(test_data, test_labels)
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    resnet_gru_model = GRUClassifier(best_resnet_model,hidden_dim, output_dim)
    LOAD_PATH = 'best_resnet_gru_model.pth'  ###output neuron is 1  ##main_output
    
    if cuda:
      resnet_gru_model.cuda()
    model_dict = resnet_gru_model.state_dict()
    pretrained_dict = torch.load(LOAD_PATH)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
   # print('weights loaded model = ', len(pretrained_dict), '/', len(model_dict))
    resnet_gru_model.load_state_dict(torch.load(LOAD_PATH))
    resnet_model_from_gru = resnet_gru_model.resnet()  
    resnet_fc_model = ResNetFCModel(resnet_model_from_gru) 
    LOAD_PATH = 'best_resnet_fc_model.pth'  ###output neuron is 1  ##main_output
    if cuda:
      resnet_fc_model.cuda()
    model_dict = resnet_fc_model.state_dict()
    pretrained_dict = torch.load(LOAD_PATH)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model=best_resnet_fc_model 
    model.eval()
    target_layer = model.module[-4] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_loader = DataLoader(torch.empty(0), batch_size=1)
# Assuming model, device, and target_layer are defined elsewhere
    pseudolabels, _ = get_pseudolabels(test_loader, val_loader, model, device, target_layer)
    model = UNet(in_channels=1, out_channels=2, init_features=16)

    LOAD_PATH = 'best_unet_model.pth'  ###output neuron is 1
    if cuda:
      model.cuda()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(LOAD_PATH)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #print('weights loaded model = ', len(pretrained_dict), '/', len(model_dict))
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()
  
    
