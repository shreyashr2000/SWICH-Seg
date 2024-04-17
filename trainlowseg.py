from augmentation import augment1
from augmentation import augment2
from windowing import window_ct
def train_model_lowdata(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
            one_d_data = torch.zeros((inputs.shape[0], 1, 512, 512)).to(device)
            for i in range(inputs.shape[0]):
                data1 = augment1(inputs[i,:,:])
                if isinstance(data1, np.ndarray):
                    gray_img = torch.from_numpy(data1)
                else:
                    gray_img = data1

                r = window_ct(gray_img, w_level=40, w_width=80)  # Extract R channel
                # Normalize the channel
                r = r / r.max()
                # Scale each channel to [0, 255] range
                r = (r * 255).clamp(0, 255)
                # Assign R, G, B channels to respective positions in the new tensor
                if not torch.isinf(r).any().item():
                    one_d_data[i, 0, :, :] = r
            inputs, targets = one_d_data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                one_d_data = torch.zeros((inputs.shape[0], 3, 512, 512)).to(device)
                for i in range(inputs.shape[0]):
                    gray_img = inputs[i,:,:]
                    r = window_ct(gray_img, w_level=40, w_width=80)  # Extract R channel
                    g = window_ct(gray_img, w_level=80, w_width=200) # Extract G channel
                    b = window_ct(gray_img, w_level=600, w_width=2800) # Extract B channel

                    # Normalize each channel individually
                    r = r / r.max()
                    g = g / g.max()
                    b = b / b.max()

                    # Scale each channel to [0, 255] range
                    r = (r * 255).clamp(0, 255)
                    g = (g * 255).clamp(0, 255)
                    b = (b * 255).clamp(0, 255)

                    # Assign R, G, B channels to respective positions in the new tensor
                    if not torch.isinf(r).any().item():
                        one_d_data[i, 0, :, :] = r
                    if not torch.isinf(g).any().item():
                        one_d_data[i, 1, :, :] = g
                    if not torch.isinf(b).any().item():
                        one_d_data[i, 2, :, :] = b

                inputs, targets = one_d_data.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, save_path)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    print('Training complete!')
    return best_model
