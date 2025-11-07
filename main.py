from trainer import train

from dataloader import mockingLoaders

from model_image_1 import initializeComponents


if __name__ == "__main__":
    train_loader, test_loader = mockingLoaders()    # CIFAR-10, 10 classes
    numClasses = 10

    model, optimizer, scheduler = initializeComponents(num_classes=numClasses)

    train(model, train_loader, test_loader, optimizer, scheduler, num_classes=numClasses)
