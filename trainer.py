import torch
import numpy as np

import config

def train(
    model,
    trainLoader,
    testLoader,
    optimizer,
    scheduler
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)  # default for binary classification

    """
    Training loop:
        Record only train/test loss for each epoch for plotting;

        Print/save loss, accuracy, F1 score for train/test only for the last epoch as final performance.
    """
    trainLossList = []
    testLossList = []
    for i in range(1, config.EPOCHS + 1):
        # Training
        model.train()
        trainLoss = 0.0
        for Xs, Ys in trainLoader:
            Xs, Ys = Xs.to(device), Ys.to(device)
            optimizer.zero_grad()
            logits = model(Xs)
            loss = criterion(logits, Ys)
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * Xs.size(0)   # total loss for this batch

        trainLossList.append(trainLoss / len(trainLoader.dataset))

        scheduler.step()

        # Testing
        model.eval()
        testLoss = 0.0
        with torch.no_grad():
            for Xs, Ys in testLoader:
                Xs, Ys = Xs.to(device), Ys.to(device)
                logits = model(Xs)
                loss = criterion(logits, Ys)

                testLoss += loss.item() * Xs.size(0)

        testLossList.append(testLoss / len(testLoader.dataset))







    # Save two lists for later plotting.
    np.savez(
        "results/two_loss_lists.npz",
        trainList=trainLossList,
        testList=testLossList
    )
