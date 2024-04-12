import pandas as pd
import matplotlib.pyplot as plt


loss_df = pd.read_csv('./loss.csv', index_col=0)
loss_df

# Plot losses
plt.figure(figsize=(10, 6))  # Set the figure size

plt.plot(loss_df['epoch'], loss_df['D_loss'], color='orange', label='Discriminator Loss')
plt.plot(loss_df['epoch'], loss_df['G_loss'], color='green', label='Generator Loss')

# Customize the plot (optional)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves (Generator vs. Discriminator)')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("loss_training.png", bbox_inches='tight')  # Adjust filename as needed


plt.show()