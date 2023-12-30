from src.models.CNN import CNN
from src.dataHandlers.imagenetteHandler import imagenetteHandler

if __name__ == "__main__":
    data = imagenetteHandler()

    cnn = CNN(
        input_shape=(None, 256, 256),
        output_shape=(None, 256, 256, 2),
        epochs=150
    )

    cnn.save("imagenet_cnn")

    cnn.train(x_train=data.x_train, y_train=data.y_train, batchSize=25, epochs=150, folder="imagenet_cnn")

    cnn.test_model("src/data/places/downloads/extracted/data/imagenette2/val/val_data/ILSVRC2012_val_00002990.JPEG")