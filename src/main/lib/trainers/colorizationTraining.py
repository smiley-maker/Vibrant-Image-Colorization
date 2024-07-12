from src.main.lib.utils.imports import *

class TrainCNN:
    def __init__(
            self,
            model,
            optimizer, 
            epochs : int, 
            batch_size : int,
            data, 
            device, 
            save_path = f"../../../reports/models/cnn",
            save_images = False,
            extra_losses = None,
        ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.extra_losses = extra_losses
        self.save_images = save_images
        self.save_path = save_path + f"-{self.epochs}epochs"
        self.writer = SummaryWriter(self.save_path)  # Create writer    
        print("Initialized trainer")

    def saturated_huber_loss(self, y_true, y_pred):
        # Both the ground truth and predicted inputs have two results for the A and B
        # color channels, respectively. So, to retrieve the values for each, we can
        # split along the 
        a_true, b_true = torch.split(y_true, split_size_or_sections=2, dim=0)
        a_pred, b_pred = torch.split(y_pred, split_size_or_sections=2, dim=0)

        # Computes the mean-squared error between the a and b values of the predicted
        # and ground truth tensors. 
        color_loss = torch.sqrt(torch.square(a_true - a_pred) + torch.square(b_true - b_pred))

        # Computes the hue and saturation from the a and b channels for both the ground truth
        # and predicted colors. 
        hue_true = torch.atan2(b_true, a_true)
        hue_pred = torch.atan2(b_pred, a_pred)
        sat_true = torch.sqrt(torch.square(a_true) + torch.square(b_true))
        sat_pred = torch.sqrt(torch.square(a_pred) + torch.square(b_pred))

        saturation_weight = 3.0 # Defines a weighting factor for emphasizing saturation over hue

        # Computes the weighted hue-saturation loss
        hue_saturation_loss = torch.square(hue_pred - hue_true) + saturation_weight * torch.square(sat_pred - sat_true)

        # Combines the color loss with the hue and saturation loss
        total_loss = torch.add(color_loss, hue_saturation_loss)

        print(torch.mean(torch.sqrt(total_loss)))

        # Return the mean loss over the batch
        return torch.mean(torch.sqrt(total_loss))

    
    def train_model(self):
        print("Training.....")
        for epoch in range(self.epochs):
            overall_loss = 0

            for batch_num, data in enumerate(self.data):
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                x = torch.reshape(x, (32, 1, 120, 120))

                self.optimizer.zero_grad()

                y_hat = self.model(x) # Breaks here
                loss = self.saturated_huber_loss(y, y_hat)

                if self.extra_losses != None:
                    for l in self.extra_losses:
                        loss += (l(y, y_hat))

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()


            # Log loss to TensorBoard
            self.writer.add_scalar("Loss/SHL", overall_loss/(batch_num*self.batch_size), epoch)

            if self.save_images:
                with torch.no_grad():
                    # Sample from latent space (replace with your sampling function)
                    sampled_z = self.sample_latent_space()
                    sampled_images = self.model.decode(sampled_z)

                    # Normalize and reshape for TensorBoard (assuming image data)
                    grid = torchvision.utils.make_grid(
                        sampled_images.detach().cpu().view(-1, *self.xdim) / 2 + 0.5, normalize=True)

                    self.writer.add_image("LatentSpace/Samples", grid, epoch)

            
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}: {overall_loss/(batch_num*self.batch_size)}")

                if self.save_images: 
                    combined_img = torch.cat((x[:4], x_hat[:4]), dim=2)
                    self.writer.add_image("Input & Colorized", vutils.make_grid(combined_img.to(self.device), normalize=True, scale=1), epoch)

        
        self.writer.close()
        return overall_loss