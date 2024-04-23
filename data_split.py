import random
from data_preprocessing import load_test_data
def data_split(img,mask,large=True,train=False):
  if train=True:
    data,mask,label=load_test_data(img,mask)
  else:
# Shuffle the list of files and masks together
    combined_data = list(zip(files, masks))
    random.shuffle(combined_data)
    files, masks = zip(*combined_data)

# Calculate the size of each fold
    fold_size = len(files) // 5

# Create an empty list to store the folds
    file_folds = []
    mask_folds = []

# Split the shuffled list of files and masks into five folds
    for i in range(0, len(files), fold_size):
        file_folds.append(files[i:i+fold_size])
        mask_folds.append(masks[i:i+fold_size])

# If there are leftover files and masks, add them to the last fold
    if len(files) % 5 != 0:
        file_folds[-1].extend(files[-(len(files) % 5):])
        mask_folds[-1].extend(masks[-(len(masks) % 5):])

# Merge the first three folds for files and masks
    train_files = file_folds[0] + file_folds[1] + file_folds[2]
    train_masks = mask_folds[0] + mask_folds[1] + mask_folds[2]

# Add the second half of the fourth fold to the training data for files and masks
    train_files += file_folds[3][half_size:]
    train_masks += mask_folds[3][half_size:]

# Use the first half of the fourth fold as validation data for files and masks
    val_files = file_folds[3][:half_size]
    val_masks = mask_folds[3][:half_size]

# Use the fifth fold as a separate test fold for files and masks
    test_files = file_folds[4]
    test_masks = mask_folds[4]
    train_data,train_mask,train_label=load_train_data(train_files,train_masks)
    val_data,val_mask,val_label=load_train_data(val_files,val_masks)
    test_data,test_mask,test_label=load_train_data(test_files,test_masks)
    if train=True:
      return train_data,train_label,val_data,val_label
    if train=False:
      return test_data,test_mask
    
