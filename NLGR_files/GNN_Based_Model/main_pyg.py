import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
#torch.autograd.detect_anomaly()
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type,logical_reg):
    model.train()
#    print([i for i in model.named_parameters()])
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
        #     pass
        # else:
        constraints,pred = model(batch,1)
        #print(constraints)
        optimizer.zero_grad()
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        if model.graph_pooling == 'logical':
            
            false_vector = model.logic_not(model.true_vector)  # we compute the representation
            
             #         here, we maximize the similarity between not true and false
            r_not_true = (1 - F.cosine_similarity(model.logic_not(model.true_vector), false_vector, dim=0))
            
            # here, we maximize the similarity between not not x and x
            r_not_not_self = (1 - F.cosine_similarity(model.logic_not(model.logic_not(constraints)), constraints)).mean()

            # here, we minimize the similarity between not x and x
            r_not_self = (1 + F.cosine_similarity(model.logic_not(constraints), constraints)).mean()

            # here, we maximize the similarity between not not true and true
            r_not_not_true = (1 - F.cosine_similarity(
                model.logic_not(model.logic_not(model.true_vector)), model.true_vector,dim=0))

   
            # And

    
            # here, we maximize the similarity between x AND True and x
            r_and_true = (1 - F.cosine_similarity(
                model.logic_and(constraints, model.true_vector.expand_as(constraints)), constraints)).mean()

            # here, we maximize the similarity between x AND False and False
            r_and_false = (1 - F.cosine_similarity(
                model.logic_and(constraints, false_vector.expand_as(constraints)),
                false_vector.expand_as(constraints))).mean()

            # here, we maximize the similarity between x AND x and x
            r_and_self = (1 - F.cosine_similarity(model.logic_and(constraints, constraints), constraints)).mean()

            # here, we maximize the similarity between x AND not x and False
            r_and_not_self = (1 - F.cosine_similarity(
                model.logic_and(constraints, model.logic_not(constraints)),
                false_vector.expand_as(constraints))).mean()

            # same rule as before, but we flipped operands
            r_and_not_self_inverse = (1 - F.cosine_similarity(
                model.logic_and(model.logic_not(constraints), constraints),
                false_vector.expand_as(constraints))).mean()
            
            

            # True/False rule
            # here, we minimize the similarity between True and False
            true_false = 1 + F.cosine_similarity(model.true_vector, false_vector, dim=0)

#             # here, we maximize similatrity between True AND True and True | R3/5 True
            true_and_true = (1 - F.cosine_similarity(model.logic_and(model.true_vector,model.true_vector,dim=0),model.true_vector, dim=0))
            # here, we maximize similatrity between True AND False and False AND True 
            true_and_false_false_and_false = (1 -F.cosine_similarity(
model.logic_and(false_vector,model.true_vector,dim=0),model.logic_and(model.true_vector,false_vector,dim=0), dim=0))
            
            
#             print(r_not_not_self,r_not_self,r_not_not_not)
            
            r_loss = (r_not_true + 
            r_not_not_self + r_not_self + r_not_not_true +
                  r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse 
#                       + true_and_true  
#                       + true_false
#                       + true_and_false_false_and_false
                     )
            #print(r_loss)
            bs = batch.y.shape[0]
            bce = torch.nn.BCELoss(reduction='sum')(pred.to(torch.float64).reshape(1,bs), batch.y.to(torch.float64).reshape(1,bs))
#            print(bce)
            loss = r_loss*logical_reg + bce
            #  cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            #print(f'reg loss is {r_loss}')
        else:
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
#        print(loss)
        loss.backward()
        optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                _,pred = model(batch,0)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    #print(y_true)
    #print(y_pred)
    #print(y_true.shape,y_pred.shape)
    return  roc_auc_score(y_true,y_pred)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--logical_reg', type=float, default=0.1,
                        help='dropout ratio (default: 0.1)')    
    parser.add_argument('--num_layer_gnn', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph pooling (default: mean)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="GLN_model",
                        help='filename to output result (default: GLN)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    graph_pooling = args.graph_pooling
    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, device = device, num_layer = args.num_layer_gnn, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling=graph_pooling ,virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, device = device, num_layer = args.num_layer_gnn, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling=graph_pooling ,virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, device = device, num_layer = args.num_layer_gnn, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling=graph_pooling ,virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, device = device, num_layer = args.num_layer_gnn, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling=graph_pooling, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    
    my_list = [
               'and_layer_2.bias', 'and_layer_2.weight',
               'and_layer_1.bias', 'and_layer_1.weight',
               'not_layer_2.bias', 'not_layer_2.weight', 
               'not_layer_1.bias', 'not_layer_1.weight'
#        ,'and_layer_3.bias', 'and_layer_3.weight'
#        ,'and_layer_4.bias', 'and_layer_4.weight'
#        ,'not_layer_3.bias', 'not_layer_3.weight'
#        ,'not_layer_4.bias', 'not_layer_4.weight'
              ]
    logical_params = (i[1] for i in list(filter(lambda kv: kv[0] in my_list, model.named_parameters())))
    params = (i[1] for i in list(filter(lambda kv: kv[0] not in my_list, model.named_parameters())))
#    print(logical_params[0][1])
    if graph_pooling == 'logical':
        optimizer = optim.Adam([{'params': params},
                {'params': logical_params, 'lr': 1e-5}
            ], lr=1e-3)
    else:
        optimizer = optim.Adam(params, lr=1e-3)
    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type,args.logical_reg)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        # train_curve.append(train_perf[dataset.eval_metric])
        # valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])

    # if 'classification' in dataset.task_type:
    #     best_val_epoch = np.argmax(np.array(valid_curve))
    #     best_train = max(train_curve)
    # else:
    #     best_val_epoch = np.argmin(np.array(valid_curve))
    #     best_train = min(train_curve)

    # print('Finished training!')
    # print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    # print('Test score: {}'.format(test_curve[best_val_epoch]))

        torch.save({'Val': valid_perf, 'Test': test_perf, 'Train': train_perf, 'BestTrain': train_perf}, "../Final_model_experiments/"+args.filename+"_"+str(args.num_layer_gnn)+"_"+str(epoch)+"_"+str(args.logical_reg)+"_"+args.graph_pooling+"_"+args.gnn+"-"+str(round(valid_perf,4))+"-"+str(round(test_perf,4)))
        torch.save(model.state_dict(),"../Final_model_experiments_models/"+args.filename+"_"+str(args.num_layer_gnn)+"_"+str(epoch)+"_"+str(args.logical_reg)+"_"+args.graph_pooling+"_"+args.gnn+"-"+str(round(valid_perf,4))+"-"+str(round(test_perf,4)))

if __name__ == "__main__":
    main()


