from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

train_losses = []
train_acc = []


def train(model, device, trainloader, epoch, reg_type):
    running_loss = 0.0
    correct = 0
    processed = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0003)
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        # Update pbar tqdm 
        output = outputs.argmax(dim=1, keepdim=True)
        correct += output.eq(labels.view_as(output)).sum().item()
        #correct += (output == labels).sum().item()
        processed += len(inputs)
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={i} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

print('Finished Training')