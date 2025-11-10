import time
import torch
import numpy as np

from utility import audio_loader_selection, audio_model_initialization


def trainForModel(model_name: str):
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_fake_class_0 = 22800
    num_real_class_1 = 2580
    total_samples = num_fake_class_0 + num_real_class_1
    weight_fake = total_samples / (2 * num_fake_class_0)
    weight_real = total_samples / (2 * num_real_class_1)
    class_weights = torch.tensor([weight_fake, weight_real]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    trainLoader, test1Loader, test2Loader = audio_loader_selection(model_name)

    model, optimizer, scheduler = audio_model_initialization(model_name, epochs)

    print(f"\nDevice: {device}, Epochs: {epochs}, Model: {model_name}")
    print(f"Class weights: Fake={weight_fake:.4f}, Real={weight_real:.4f}")
    print(f"{'='*80}")

    model = model.to(device)
    
    # Storage for metrics
    train_losses = []
    train_accs = []
    test1_losses = []
    test1_accs = []
    test2_losses = []
    test2_accs = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for Xs, ys in trainLoader:
            Xs = Xs.to(device)
            ys = ys.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(Xs)
            loss = criterion(outputs, ys)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * Xs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == ys).sum().item()
            train_total += ys.size(0)
        
        # Calculate training metrics
        assert train_total == num_fake_class_0 + num_real_class_1
        avg_train_loss = train_loss / train_total
        train_accuracy = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        
        # Evaluate on Test Set 1
        model.eval()
        test1_loss = 0.0
        test1_correct = 0
        test1_total = 0
        
        with torch.no_grad():
            for Xs, ys in test1Loader:
                Xs = Xs.to(device)
                ys = ys.to(device)
                
                outputs = model(Xs)
                loss = criterion(outputs, ys)
                
                test1_loss += loss.item() * Xs.size(0)
                _, predicted = torch.max(outputs, 1)
                test1_correct += (predicted == ys).sum().item()
                test1_total += ys.size(0)
        
        avg_test1_loss = test1_loss / test1_total
        test1_accuracy = 100.0 * test1_correct / test1_total
        test1_losses.append(avg_test1_loss)
        test1_accs.append(test1_accuracy)
        
        # Evaluate on Test Set 2
        test2_loss = 0.0
        test2_correct = 0
        test2_total = 0
        
        with torch.no_grad():
            for Xs, ys in test2Loader:
                Xs = Xs.to(device)
                ys = ys.to(device)
                
                outputs = model(Xs)
                loss = criterion(outputs, ys)
                
                test2_loss += loss.item() * Xs.size(0)
                _, predicted = torch.max(outputs, 1)
                test2_correct += (predicted == ys).sum().item()
                test2_total += ys.size(0)
        
        avg_test2_loss = test2_loss / test2_total
        test2_accuracy = 100.0 * test2_correct / test2_total
        test2_losses.append(avg_test2_loss)
        test2_accs.append(test2_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch statistics in one line
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}% | Test1 Loss: {avg_test1_loss:.4f} Acc: {test1_accuracy:.2f}% | Test2 Loss: {avg_test2_loss:.4f} Acc: {test2_accuracy:.2f}% | Time: {epoch_time:.2f}s")
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Final Train      - Loss: {train_losses[-1]:.4f} | Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Test Set 1 - Loss: {test1_losses[-1]:.4f} | Accuracy: {test1_accs[-1]:.2f}%")
    print(f"Final Test Set 2 - Loss: {test2_losses[-1]:.4f} | Accuracy: {test2_accs[-1]:.2f}%")
    print(f"{'='*80}\n")
    
    # Save results to NPZ file
    results = {
        'train': train_losses,
        'test_1': test1_losses,
        'test_2': test2_losses
    }
    
    filename = f'results/{model_name}_losses.npz'
    np.savez(filename, **results)
    print(f"Results saved to {filename}")
