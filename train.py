import time
from tqdm import tqdm

def evaluate(net, test_loader, device="cpu"):
    net.eval() 
    total_num =0
    num_ac = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred_y = net(X)
            num_ac += sum(pred_y.argmax(1) == y)
            total_num += y.numel()
    return num_ac/total_num
                


def train(net, loss, optimizer, train_loader, test_loader, print_loss=False, num_epochs=100, device="cpu"):
    net.to(device)
    time_start = time.time()
    for epoch in range(num_epochs):
        num_ac = 0
        num_train = 0 
        net.train()
        for i, (X, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            loss_val = l.item()
            l.backward()
            optimizer.step()
            num_ac += sum(y_hat.argmax(1) == y)
            num_train += y.numel()
            if print_loss:
                print(f'epoch {epoch+1} batch {i+1}, loss {loss_val:.3f}')
        print(f'epoch {epoch+1}, loss {loss_val:.3f}, acc {(num_ac/num_train):.3f}')
        
    time_end = time.time()
    print("train finish!\ncomputing result...")
    test_acc = evaluate(net, test_loader, device)             
    print(f'loss {loss_val:.3f}, train acc {(num_ac/num_train):.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{(num_train / (time_end-time_start)):.1f} examples/sec '
          f'on {str(device)}')
