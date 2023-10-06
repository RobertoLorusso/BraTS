from brats22_model import *
from loss import *

# Initialize the model
autoencoder = UNet3D()

# Define loss function and optimizer
criterion = LossBraTS()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Simulated 3D image data (batch_size, channels, depth, height, width)
# Replace this with your actual data loading code
input_data = torch.randn(16, 1, 32, 32, 32)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()

    outputs = autoencoder(input_data)
    loss = criterion(outputs, input_data)

    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
