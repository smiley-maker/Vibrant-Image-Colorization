#from src.main.lib.models.CNN import CNN
from src.main.lib.utils.imports import *
from src.main.lib.models.colorization_model import CNN
from src.main.lib.dataHandlers.pokemonHandler import pokemonHandler, PokemonDataset
from src.main.lib.trainers.colorizationTraining import TrainCNN

if __name__ == "__main__":    
    # Creates a pytorch device
    device = torch.device("mps")

    # Settings
    batch_size = 32
    num_epochs = 500

    # Creates the initial model
    model = CNN(
        batch_size=batch_size,
        num_layers=10,
        inp_size=(1, 256, 256),
        leaky_relu=True,
        batch_norm=True,
        device=device
    )

    # sets up a pokemonHandler object
    dataset = PokemonDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)#, collate_fn=collate)

    # Creates an optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Sets up the trainer
    trainer = TrainCNN(
        model=model,
        optimizer=optimizer,
        epochs=num_epochs,
        batch_size=batch_size,
        data=dataloader,
        device=device,
        save_images=True
    )

    print(f"Created trainer, training for {num_epochs} epochs...")
    trainer.train_model()

    # Testing model
    test_x, test_y, t = dataset[4]

    pred_y = model(test_x)

    print(f"Loss for sample prediction: {trainer.saturated_huber_loss(test_y, pred_y)}")
    pred_y = pred_y * 128
    pred_y = pred_y.reshape(256, 256, 2)
    lab_image = cv2.merge((test_x, pred_y[:,:,0], pred_y[:,:,1]))
    bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    lab_image_gt = cv2.merge((test_x, test_y[:,:,0], test_y[:,:,1]))
    bgr_image_gt = cv2.cvtColor(lab_image_gt, cv2.COLOR_LAB2BGR)

    # Display the final images
    cv2.imshow("Combined Image", bgr_image)
    cv2.imshow("Original Image", bgr_image_gt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 
#    handler = pokemonHandler()
#    handler.prepareAllInputData()
#    handler.collectData()

#    print(f"training data has {len(handler.xTrain)} samples")
#    print(f"y train has shape: {handler.yTrain.shape}")

    # Creating a CNN object
#    cnn = CNN(0.001)

#    cnn.train(handler.xTrain, handler.yTrain, batchSize=32, epochs=10, folder="reports/", callBacks=[], lr=0.001)

#    cnn.model.fit(handler.xTrain, handler.yTrain, batch_size=handler.BATCH_SIZE)