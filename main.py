from trainer import train

from dataloader import imageFFLoaders

from model_image_2 import initializeComponents


if __name__ == "__main__":
    train_loader, test_loader = imageFFLoaders()
    numClasses = 6

    model, optimizer, scheduler = initializeComponents(num_classes=numClasses, image_size=128)

    train(model, train_loader, test_loader, optimizer, scheduler, num_classes=numClasses)
