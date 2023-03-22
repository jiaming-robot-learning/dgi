
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import numpy as np

from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import GIN
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from net import DGI, RegressionModel, Critic
from util import cycle_index



def pretrain_gnn(gnn, device, loader, args,writer):
    critic = Critic(args['embed_dim'])
    model = DGI(gnn, critic)
    model.to(device)

    #set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'] )
    model.train()

    for epoch in range(args['epochs_pretrain']):
        train_acc_accum = 0
        train_loss_accum = 0
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            node_ft = model.gnn(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            graph_ft = model.pool(node_ft,batch.batch)

            positive_expanded_graph_ft = graph_ft[batch.batch]

            shifted_graph_ft = graph_ft[cycle_index(len(graph_ft), 1)]
            negative_expanded_graph_ft = shifted_graph_ft[batch.batch]

            positive_score = model.critic(node_ft, positive_expanded_graph_ft)
            negative_score = model.critic(node_ft, negative_expanded_graph_ft)      

            optimizer.zero_grad()
            loss = model.loss(positive_score, torch.ones_like(positive_score)) + model.loss(negative_score, torch.zeros_like(negative_score))
            loss.backward()

            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
            train_acc_accum += float(acc.detach().cpu().item())

        train_acc, train_loss = train_acc_accum/(step+1), train_loss_accum/(step+1)

        print(f'Epoch {epoch}, train acc: {train_acc:.2%}, train loss: {train_loss:.2f}')
        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('train/acc',train_acc,epoch)

        # if evaluate
        # eval_acc, eval_loss = eval(model,device,loader_val)
        # print(f'Epoch {epoch}, eval acc: {eval_acc:.2%}, eval loss: {eval_loss:.2f}')
        # writer.add_scalar('eval/loss',eval_loss,epoch)
        # writer.add_scalar('eval/acc',eval_acc,epoch)

    return gnn

def eval(model,device,loader):
    """
    Evaluate DGI where training
    """
    model.eval()
    with torch.no_grad():

        eval_acc_accum = 0
        eval_loss_accum = 0

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            node_ft = model.gnn(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            graph_ft = model.pool(node_ft,batch.batch)

            positive_expanded_graph_ft = graph_ft[batch.batch]

            shifted_graph_ft = graph_ft[cycle_index(len(graph_ft), 1)]
            negative_expanded_graph_ft = shifted_graph_ft[batch.batch]

            positive_score = model.critic(node_ft, positive_expanded_graph_ft)
            negative_score = model.critic(node_ft, negative_expanded_graph_ft)      

            loss = model.loss(positive_score, torch.ones_like(positive_score)) \
                        + model.loss(negative_score, torch.zeros_like(negative_score))

            eval_loss_accum += float(loss.detach().cpu().item())
            acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0))\
                .to(torch.float32)/float(2*len(positive_score))
            eval_acc_accum += float(acc.detach().cpu().item())

        return eval_acc_accum/(step+1), eval_loss_accum/(step+1)

def finetune(args,gnn,device,dataset,writer):

    net = RegressionModel(args['embed_dim'],1,gnn,info_max=args['finetune_contrastive_loss'])
    optm = torch.optim.Adam(net.parameters(),lr=args['reg_lr'])
    split = int(len(dataset)*0.8)
    loader_train = DataLoader(dataset[:split], batch_size=args["batch_size"], shuffle=True, num_workers = 4)
    loader_eval = DataLoader(dataset[split:], batch_size=args["batch_size"], shuffle=True, num_workers = 4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    net.to(device)

    for ep in range(args["epochs_finetune"]):
        for step, batch in enumerate(loader_train):
            batch = batch.to(device)

            pred, node_ft, graph_ft = net(batch)
            pred = pred.squeeze()

            if args['finetune_contrastive_loss']:
                loss_reg = loss_fn(pred,batch.y.squeeze())

                positive_expanded_graph_ft = graph_ft[batch.batch]
                shifted_graph_ft = graph_ft[cycle_index(len(graph_ft), 1)]
                negative_expanded_graph_ft = shifted_graph_ft[batch.batch]

                positive_score = net.critic(node_ft, positive_expanded_graph_ft)
                negative_score = net.critic(node_ft, negative_expanded_graph_ft)      

                loss_info_max = loss_fn(positive_score, torch.ones_like(positive_score)) + loss_fn(negative_score, torch.zeros_like(negative_score))
                loss = 0.5*loss_reg + loss_info_max

            else:
                loss = loss_fn(pred,batch.y.squeeze())

            optm.zero_grad()
            loss.backward()
            optm.step()

            loss_train = loss.detach().cpu().item()

        print(f'Fine tune ep {ep}: train loss: {loss_train:.3f}')
        writer.add_scalar('finetune/loss',loss_train,ep)

    print(f'Finished fine tunning...')

    print(f'Now evaluation...')

    pred_all = []
    y_all = []
    net.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader_eval):
            batch = batch.to(device)
            # node_ft = model.gnn(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            # graph_ft = model.pool(node_ft,batch.batch)

            pred,_,_ = net(batch)
            pred = torch.sigmoid(pred)
            pred_all.append(pred.detach().cpu())
            y_all.append(batch.y.cpu())


        pred = torch.cat(pred_all,dim=0).squeeze(1)
        y = torch.cat(y_all,dim=0).squeeze(1)

        # eval metrics
        precision, recall, thresholds = precision_recall_curve(y, pred)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)

        roc = roc_auc_score(y,pred)

    return auc_precision_recall, roc

def main(args):
    print('======== new exp starts =======')
    for k,v in args.items():
        print(f'{k}: {v}')
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    device = torch.device("cuda:1")
    torch.cuda.manual_seed_all(args['seed'])

    dataset = MoleculeNet(
                        root='data/molecular_net',
                        name='HIV',
    )

    ys = [0,0]
    t_n = 0
    node_n = 0
    edge_n = 0
    for i, d in enumerate(dataset):
        ys[int(d.y.item())] += 1
        t_n += 1
        node_n += d.x.shape[0]
        edge_n += d.edge_index.shape[1] // 2

    print(f'Total number of graphs in dataset: {t_n}.')
    print(f'Num of positive labels: {ys[1]} - {ys[1]/t_n:.2%}')
    print(f'Num of negative labels: {ys[0]} - {ys[0]/t_n:.2%}')
    print(f'Avg. num of nodes per graph: {node_n/t_n:.2f}')
    print(f'Avg. num of edges per graph: {edge_n/t_n:.2f}')
    print(f'Num of node features: {d.x.shape[1]}')
    print(f'Num of edge features: {d.edge_attr.shape[1]}')

    
    if not args['use_whole_dataset']:
        pre_train_split = int(t_n*0.7) # assumes we only have labels for 30% of the data
        loader_pretrain = DataLoader(dataset[:pre_train_split], batch_size=args["batch_size"], shuffle=True, num_workers = 4)
        finetune_dataset = dataset[pre_train_split:]
    else:
        loader_pretrain = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers = 4)
        finetune_dataset = dataset

    gnn = GIN(
        in_channels=d.x.shape[1],
        hidden_channels=args['embed_dim'],
        num_layers=args['num_layers']
    )

    writer = SummaryWriter(f'runs/{args["exp_name"]}_{args["seed"]}')

    if args['pretrain']:
        print(f'Pretraining...')
        gnn = pretrain_gnn(gnn,device,loader_pretrain,args,writer)
        # freeze gnn layers
        for p in gnn.parameters():
            p.requires_grad = False


    pr_aoc, roc = finetune(args,gnn,device,finetune_dataset,writer)
    print(f'Final PR_AOC on eval set: {pr_aoc:.4f}.')
    print(f'Final ROC on eval set: {roc:.4f}.')
    writer.add_scalar('pr-aoc',pr_aoc)
    writer.add_scalar('roc',roc)

    return pr_aoc, roc

if __name__ == "__main__":

    results = [[] for _ in range (6)]
    for seed in range(2):
        # pretrain
        args = {
            "batch_size": 128,
            "epochs_pretrain": 50,
            "lr": 1e-3,
            'num_layers': 2,
            'embed_dim': 300,
            "reg_lr":1e-3,
            "epochs_finetune": 10,
            "pretrain" : True,
            "use_whole_dataset" : True,
            'exp_name': 'pretrain-whole',
            'seed': seed,
            "finetune_contrastive_loss":False,
        }
        results[0].append(main(args))
        args = {
            "batch_size": 128,
            "epochs_pretrain": 50,
            "lr": 1e-3,
            'num_layers': 2,
            'embed_dim': 300,
            "reg_lr":1e-3,
            "epochs_finetune": 10,
            "pretrain" : True,
            "use_whole_dataset" : False,
            'exp_name': 'pretrain-partial',
            'seed': seed,
            "finetune_contrastive_loss":False,
        }
        results[1].append(main(args))

        # not pretrain
        args = {
            "batch_size": 128,
            "epochs_pretrain": 50,
            "lr": 1e-3,
            'num_layers': 2,
            'embed_dim': 300,
            "reg_lr":1e-3,
            "epochs_finetune": 50,
            "pretrain" : False,
            "use_whole_dataset" : True,
            'exp_name': 'no-pretrain-whole',
            'seed': seed,
            "finetune_contrastive_loss":False,
        }
        results[2].append(main(args))

        args = {
            "batch_size": 128,
            "epochs_pretrain": 50,
            "lr": 1e-3,
            'num_layers': 2,
            'embed_dim': 300,
            "reg_lr":1e-3,
            "epochs_finetune": 50,
            "pretrain" : False,
            "use_whole_dataset" : False,
            'exp_name': 'no-pretrain-partial',
            'seed': seed,
            "finetune_contrastive_loss":False,
        }
        results[3].append(main(args))

        # not pretrain, combine info max loss
        args = {
            "batch_size": 128,
            "epochs_pretrain": 50,
            "lr": 1e-3,
            'num_layers': 2,
            'embed_dim': 300,
            "reg_lr":1e-3,
            "epochs_finetune": 50,
            "pretrain" : False,
            "use_whole_dataset" : True,
            'exp_name': 'no-pretrain-whole-combine-loss',
            'seed': seed,
            "finetune_contrastive_loss":True,
        }
        results[4].append(main(args))

        args = {
            "batch_size": 128,
            "epochs_pretrain": 50,
            "lr": 1e-3,
            'num_layers': 2,
            'embed_dim': 300,
            "reg_lr":1e-3,
            "epochs_finetune": 50,
            "pretrain" : False,
            "use_whole_dataset" : False,
            'exp_name': 'no-pretrain-partial-combine-loss',
            'seed': seed,
            "finetune_contrastive_loss":True,
        }
        results[5].append(main(args))

    pr_aoc, roc = [], []
    for i in range(len(results)):
        exp_result = results[i]
        if len(exp_result)==0:
            continue
        p, r = 0, 0
        for v1, v2 in exp_result:
            p += v1
            r += v2
        p = p / len(exp_result)
        r = r / len(exp_result)
        pr_aoc.append(p)
        roc.append(r)
    print(pr_aoc)
    print(roc)
