import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score

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

        Print/save loss, accuracy, precision, F1 score for train/test only for the last epoch as final performance.
    """
    isLastEpoch = False
    trainLossList = []
    testLossList = []

    trainPreds = []
    trainTargets = []
    testPreds = []
    testTargets = []
    
    for i in range(1, config.EPOCHS + 1):
        if i == config.EPOCHS:
            isLastEpoch = True

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
            
            if isLastEpoch:
                preds = torch.argmax(logits, dim=1)
                trainPreds.extend(preds.cpu().numpy())
                trainTargets.extend(Ys.cpu().numpy())

        trainLossList.append(trainLoss / len(trainLoader.dataset))

        if scheduler is not None:
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
                
                if isLastEpoch:
                    preds = torch.argmax(logits, dim=1)
                    testPreds.extend(preds.cpu().numpy())
                    testTargets.extend(Ys.cpu().numpy())

        testLossList.append(testLoss / len(testLoader.dataset))

        # Print loss for this epoch all in one line
        print(
            f"Epoch {i:03d}/{config.EPOCHS} | "
            f"Train Loss: {trainLossList[-1]:.4f} | "
            f"Test Loss: {testLossList[-1]:.4f}"
        )
        
    # Print final metrics
    trainAcc = accuracy_score(trainTargets, trainPreds)
    trainPrec = precision_score(trainTargets, trainPreds, average='binary')
    trainF1 = f1_score(trainTargets, trainPreds, average='binary')
    
    testAcc = accuracy_score(testTargets, testPreds)
    testPrec = precision_score(testTargets, testPreds, average='binary')
    testF1 = f1_score(testTargets, testPreds, average='binary')
    
    print("\n" + "="*70)
    print(f"FINAL EPOCH ({config.EPOCHS}) RESULTS:")
    print("="*70)
    print(f"Training Set:")
    print(f"  Loss: {trainLossList[-1]:.4f} | Accuracy: {trainAcc:.4f} | Precision: {trainPrec:.4f} | F1: {trainF1:.4f}")
    print(f"\nTest Set:")
    print(f"  Loss: {testLossList[-1]:.4f} | Accuracy: {testAcc:.4f} | Precision: {testPrec:.4f} | F1: {testF1:.4f}")
    print("="*70 + "\n")


    # Save two lists for later plotting.
    np.savez(
        "results/two_loss_lists.npz",
        trainList=trainLossList,
        testList=testLossList
    )
