import pandas as pd
import matplotlib.pyplot as plt


name_file = "optimized_model"
# Load the CSV file
csv_file = 'ProposedNetwork_StratifiedKFold_Final_fold_5_training.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file)

# Extract the required columns
epochs = range(1, len(data) + 1)  # Epoch numbers
train_acc = data['accuracy']      # Training accuracy
val_acc = data['val_accuracy']    # Validation accuracy
train_loss = data['loss']         # Training loss
val_loss = data['val_loss']       # Validation loss

plt.figure(figsize=(15, 10)) 
plt.plot(epochs,train_loss,'r', label='Train_loss')
plt.plot(epochs,val_loss,'b', label='Val_loss')
plt.title('Train Loss vs Validation Loss', fontsize=24)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16) 
plt.savefig("./Plot/" +name_file+'_Loss.png',dpi=300)
#plt.figure()
plt.show()

plt.figure(figsize=(15, 10)) 
plt.plot(epochs,train_acc,'r', label='Train_acc')
plt.plot(epochs,val_acc,'b', label='Val_acc')
plt.title('Train Accuracy vs Validation Accuracy', fontsize=24)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16) 
plt.savefig("./Plot/" +name_file+'_Acc.png',dpi=300)
#plt.figure()
plt.show()
