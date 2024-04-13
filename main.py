import os
import data
import models
import options
import plot
import csv
import ssl
from datetime import datetime


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    opt = options.parse_args_train()
    
    print(opt)
    with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
        f.write(str(opt) + '\n')
    with open(os.path.join(opt.checkpoint_dir, 'loss_np'), 'w') as f:
        f.write('iter,loss\n')
    dataloader = data.get_dataloader(True, opt.batch, opt.dataset_dir)
    model = models.ResNetModel(opt, train=True)
    
    total_iter = 0
    loss = 0.0
    training = True
    while training:
        for inputs, labels in dataloader:
            total_iter += 1
            model.optimize_params(inputs, labels)
            loss += model.get_current_loss()
            if total_iter % opt.print_freq == 0:
                txt = f'iter: {total_iter: 6d}, loss: {loss / opt.print_freq}'
                print(txt)
                with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
                    f.write(txt + '\n')
                with open(os.path.join(opt.checkpoint_dir, 'loss_np'), 'a') as f:
                    f.write(f'{total_iter},{loss / opt.print_freq}\n')
                loss = 0.0
                
            if total_iter % opt.save_params_freq == 0:
                model.save_model(f'{total_iter // opt.save_params_freq}k')
                
            if total_iter == opt.n_iter:
                model.save_model('final')
                fname = 'loss_np'
                plot.plot_chart(opt.checkpoint_dir, fname)
                training = False
                break
                
    # test
    total_n = 0
    total_correct = 0
    test_loss = 0
    test_losses = [] 
    dataloader = data.get_dataloader(False, opt.batch, opt.dataset_dir)
    ts = int(datetime.timestamp(datetime.now()))
    fname = f'test_loss_np_{ts}'
    open(f'{opt.checkpoint_dir}/test_loss_log_{ts}.txt', 'w')
    with open(f'{opt.checkpoint_dir}/{fname}', 'w') as f:
        f.write('iter,loss\n')
        
    for batch in dataloader:
        inputs, labels = batch
        correct, total, pred_y, outputs = model.test(inputs, labels)
        total_correct += correct
        total_n += total
        loss = model.criterion(outputs, labels)
        
        test_loss += loss.item()
        test_losses.append(loss)
        
        txt = f'iter: {total_n: 4d}, loss: {loss} / {test_loss}'
        print(txt)
        with open(f'{opt.checkpoint_dir}/test_loss_log_{ts}.txt', 'a') as f:
            f.write(txt + '\n')
        with open(f'{opt.checkpoint_dir}/{fname}', 'a') as f:
            f.write(f'{total_n},{loss}\n')
            
    acc = 100 * total_correct / total_n
    err = 100 - acc
    print(f'accuracy: {acc:.2f} %')
    print(f'error: {err:.2f} %')
    print(f'{total_correct} / {total_n}')
    
    plot.plot_chart(opt.checkpoint_dir, fname, lname='Test Loss')
    plot.plot_chart(opt.checkpoint_dir, 'loss_np', fname)
    
    
    # kaggle output
    pred_ys = []
    dataloader = data.get_kaggle_dataloader(opt.batch, opt.dataset_dir)
    for batch in dataloader:
        pred_ys.append(model.pred(batch))
        
    # Open a new CSV file in write mode
    with open('yanbing_output_kaggle.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['ID', 'Labels'])
        # Write the labels array with IDs
        idk = 0
        ids = []
        lid = [8, 2, 9,0,4,3,6,1,7,5]
        for ys in pred_ys:
            for y in ys:
                j = idk // 1000
                if lid[j] != y:
                    ids.append(idk)
                    
                writer.writerow([idk, y.item()])
                idk += 1
        print(f'kaggle done', 1-len(ids)/10000.0)
        
#       data = np.genfromtxt('yanbing_output_kaggle.csv', delimiter=',', skip_header=1)
#       
#       # 'data' is now a NumPy array containing the data from the CSV file
#       for i in data:
#           for j,l in enumerate(lid):
#               if j*1000 <= i[0] < j*1000+999:
#                   if i[1] != l: 
#                       ids.append(int(i[0]))
#       # print(ids)
#       print(1-len(ids)/10000.0)