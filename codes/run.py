import os, sys, argparse, torch, random, json, glob, pickle, re, tqdm 
from os.path import abspath, dirname, join, exists
from os import makedirs
from sys import stdout
import numpy as np
from copy import deepcopy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from model import Model
from event_dataset import Event_Dataset
from collate import collate
from constants import ind2label, label2ind
from evaluator import Evaluator
EVA = Evaluator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
soft = torch.nn.Softmax(dim=1)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def predict(model, test_dataloader):
    # use a trained model to make predictions over data
    model = model.to(device)
    model.eval()
    
    Y = []
    pred_Y = []
    events = []
    pred_probs = []
    
    with torch.no_grad():      
        num_batches = len(test_dataloader)         
        for batch_item in tqdm.tqdm(test_dataloader, desc = "Making predcitions"):

            logits = model.forward(batch_item)[0]
            probs = soft(logits).cpu().tolist()
            pred_probs.extend(probs)
            pred_Y.extend(np.argmax(probs, 1))
            Y.extend(batch_item['labels'].tolist())
            events.extend(batch_item["events"])
    return events, pred_probs, pred_Y, Y


def evaluate(model, val_dataloader):
    # evaluate a model over labeled data 
    print(" >>> EVALUATION")
    events, pred_probs, y_pred, y_true = predict(model, val_dataloader)

    individual_eval, total_eval = EVA.eval(y_true, y_pred, ind2label)
    f1 = total_eval[-1]

    eval_str,_ = EVA.print_eval((individual_eval, total_eval), ind2label, None)
    cnf_mat_str = EVA.text_confusion_matrix(y_true, y_pred, ind2label)

    return eval_str, cnf_mat_str, f1 

def load_train_vars(model_dir):
    train_vars_file = join(model_dir, 'train_config.json')
    with open(train_vars_file) as f:
        train_vars = json.load(f)
    return train_vars

def test_model(model_dir, test_data_infile):
    # test a model over labeled test data. This includes loading the model and loading the data
    train_vars = load_train_vars(model_dir)
    model_type = train_vars['model_type']
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    model = Model(model_type)
    model_file = join(model_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_file))

    print(f"Making test dataset ")
    with open(test_data_infile, "r") as f:
        event2label = json.load(f)
    events = list(event2label.keys())
    labels = [ event2label[e] for e in events]
    input_ids = [tokenizer.encode(e) for e in events]
    data = {
        "events":events, 
        "labels": labels , # hard labels, each item is a label_str 
        "input_ids": input_ids
        }
    dataset = Event_Dataset(data, shuffle = False)
    test_dataloader = DataLoader(dataset, batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
    return evaluate(model, test_dataloader)


def train(model, train_dataloader, val_dataloader, args, ith_iter = None):
    # train a model over train data
    model_save_dir = args.model_save_dir
    if not exists(model_save_dir):
        makedirs(model_save_dir)
    model_file = join(model_save_dir, "model.ckpt")
    num_batches = len(train_dataloader)
    
    model.to(device)
    """ optimizer """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
               {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
               {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]
    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr, correct_bias=False)

    """ scheduler """
    num_training_steps = len(train_dataloader) * args.epoch
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    best_f1 = None
    loss_funct = torch.nn.CrossEntropyLoss()
    eval_str = ""
    for epoch in tqdm.tqdm(range(args.epoch), desc = f"Epochs {f'of {ith_iter}th iteration' if ith_iter is not None else ''}"):
        avg_loss = 0
        model.train()
        for batch_i, train_item in tqdm.tqdm(enumerate(train_dataloader), total = len(train_dataloader), desc = "Training batch"):
            logits = model.forward(train_item)[0]
            loss = loss_funct(logits, train_item["labels"].to(device)) # batch size
            avg_loss += float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_loss /= num_batches
        print(f"EPOCH {epoch} average training loss: {avg_loss}")

        if val_dataloader:
            print(f"----Evaluation----")
            cur_eval_str, cnf_mat_str, f1 = evaluate(model, val_dataloader)
            print(f"Previous best F1 = {best_f1 if best_f1 is not None else 'None'}")

            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                eval_str = cur_eval_str
                torch.save(model.state_dict(), model_file)
        else:
            torch.save(model.state_dict(), model_file)
            
        print()
    
    train_var_file = join(model_save_dir, 'train_config.json')
    with open(train_var_file, 'w') as f:
        json.dump(vars(args), f)

    print(f"Model saved to {model_file}. Training config saved to {train_var_file}")
    return model_file, eval_str



def do_training(args):
    # function to run DEST and train a model
    if not exists(args.model_save_dir):
        makedirs(args.model_save_dir)
    """ ================ make dataset ================ """
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    if not exists(f'./temp/{args.model_type}_e2id.json'):
        e2id = {} # map event to input ids
    else:
        print("Loading e2id from cache")
        e2id = json.load(open(f'./temp/{args.model_type}_e2id.json'))

    senti_polar = json.load(open(args.senti2polar)) # sentiment expression to polarity score 
    senti2polar = {}
    for senti, polar in senti_polar:
        senti2polar[senti] = [float(i) for i in polar]

    print("Making training data")
    with open(join(args.train_dir, 'labeled_events.json')) as f:
        levent2label = json.load(f) # map labeled event 2 label in the training set 
        assert len(levent2label)
        for e in levent2label:
            if e not in e2id:
                e2id[e] = tokenizer.encode(e)
    print(f"{len(levent2label)} events in the training set")
    
    dataloaders = {
        "dev":None,
        'unlabel': None,
    }
    if args.dev_dir is not None:         
        print(f"Making dev dataset ")
        with open(join(args.dev_dir, "labeled_events.json"), "r") as f:
            event2label = json.load(f)
        events = list(event2label.keys())
        labels = [ event2label[e] for e in events]
        for e in events:
            if e not in e2id:
                e2id[e] = tokenizer.encode(e)
        input_ids = [e2id[e] for e in events]
        data = {
            "events":events, 
            "labels": labels , # hard labels, each item is a label_str 
            "input_ids": input_ids
            }
        dataset = Event_Dataset(data, shuffle = False)
        dataloaders['dev'] = DataLoader(dataset, batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
        print(f"{len(events)} events in the development set")
    
    #unlabeled dataset
    uevent2polar = {}
    if args.unlabeled_event2sentis:
        uevent2sentis = json.load(open(args.unlabeled_event2sentis))
        for e in uevent2sentis:
            if e in levent2label:
                continue 
            polars = [] 
            for senti in set(uevent2sentis[e]):
                polars.append(senti2polar[senti])
            avg_polar = np.mean(polars, axis = 0)
            uevent2polar[e] = avg_polar.tolist()

        missing_uevents = [e for e in uevent2polar if e not in e2id]
        for e in tqdm.tqdm(missing_uevents, desc = "Encoding unlabeled events"):
            if e not in e2id:
                e2id[e] = tokenizer.encode(e)
    
    if not exists('./temp'):
        os.makedirs('./temp')
    if not exists(f'./temp/{args.model_type}_e2id.json'):
        print("Saving e2id to cache")
        with open(f'./temp/{args.model_type}_e2id.json', 'w') as f:
            json.dump(e2id, f)

    """ ================ train ================ """    
    print("Start training ")
    eval_log = open(join(args.model_save_dir, 'eval_result_during_training.txt'), 'w')
    model_file = None
    for cyc in tqdm.tqdm(range(args.iter), desc = "DEST CYCLE"):

        levents = list(sorted(levent2label.keys()))
        data = {
            "events":levents, 
            "labels": [levent2label[e] for e in levents], # hard labels, each item is a label_str 
            "input_ids": [e2id[e] for e in levents]
        }
        trainloader = DataLoader(Event_Dataset(data), batch_size = args.batch_size, shuffle = args.to_shuffle, collate_fn = collate)

        model = Model(args.model_type)
        model_file, dev_eval_str = train(model, trainloader, dataloaders["dev"], args, ith_iter = cyc )
        del model 
        torch.cuda.empty_cache()

        if dev_eval_str is not None:
            eval_log.write(f"CYC: {cyc}\nDEV\n{dev_eval_str}\n\n")
        
        count = 0 
        if cyc != args.iter - 1:
            print("\nPredict over unlabeled data")
            model = Model(args.model_type)
            model.load_state_dict(torch.load(model_file))

            # predict on unlabeled events
            uevent_list = list(uevent2polar.keys())
            input_ids = [e2id[e] for e in uevent_list]
            data = {
            "events":uevent_list, 
            "input_ids": input_ids
            }
            uloader = DataLoader(Event_Dataset(data),batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
            
            pred_events, pred_probs, pred_Y, _ = predict(model, uloader)
            for e, pred_prob, pred in zip(pred_events, pred_probs, pred_Y):
                label = ind2label[pred]

                joint_prob = np.multiply(pred_prob, uevent2polar[e])
                joint_prob = joint_prob/np.sum(joint_prob)

                if joint_prob[pred] >= args.threshold :
                    levent2label[e] = label
                    uevent2polar.pop(e)
                    count += 1
                elif pred_prob[2] >= args.neu_threshold:
                    levent2label[e] = 'neu'
                    uevent2polar.pop(e)

            print(f"CYC {cyc}, learned {count} new events, {len(uevent2polar)} left in the unlabeled events")
            del model 
            torch.cuda.empty_cache()
        if count == 0:
            print(f"Tranining stops at cyc {cyc}.")
            break 

    # eval over test data
    if model_file is not None and args.test_dir:
        test_eval_str, test_cnf_mat_str, _ = test_model(dirname(model_file), join(args.test_dir, "labeled_events.json"))
        eval_log.write(f"TEST\n{test_eval_str}\n{test_cnf_mat_str}\n\n")

    eval_log.close()



def get_args():
    """ ================= parse =============== """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action = 'store_true', help = "If True, train a model")
    parser.add_argument("--test", action = 'store_true', help = 'If True, test a trained model over the testing data (labeled_events.json)')
    parser.add_argument("--predict", action = 'store_true', help = 'If True, predict over unseen data with a trained model')
    parser.add_argument("--checkpoint_dir", default = None, help = 'Directory that contains a model checkpoint to be used for the prediction or testing mode')

    # ------- argments for training or testing a DEST model. The following arguments would be used if --train is True ------
    parser.add_argument("--dev_dir", default = None, help = "dir that contains preprocessed labeled_events.json for development")
    parser.add_argument("--train_dir",default = None, help = "dir that contains preprocessed labeled_events.json for training")
    parser.add_argument("--test_dir", default = None, help = "dir that contains preprocessed labeled_events.json for testing")

    parser.add_argument("--unlabeled_event2sentis", default = "data/unlabeled_data/unlabeled_event2sentis.json", help = "the file containing a dictionary mapping an unlabeled event (key) to the corresponding set of coreferent sentiment expressions (value)")
    parser.add_argument("--senti2polar",  default = "data/unlabeled_data/senti2polar.json", help = "the file containing a dictionary that maps a sentiment expression (key) to the polarity score vector (values for positive, negative and neutral) produced by a sentiment classifier")
    parser.add_argument("--threshold", default = 0.95, type = float, help = "threshold used to select newly labeled event")
    parser.add_argument("--neu_threshold", default = 0.9, type = float, help = "threshold used to specificially select extra new neutral events")
    parser.add_argument("--iter", default = 10, type = int, help = "Maximum number of iterations of discourse-enhanced self-training")
    parser.add_argument("--test_batch_size", default = 100, type = int, help = "batch size during testing a model")
    parser.add_argument("--dropout", type = float, default = 0)
    parser.add_argument("--model_type", default = 'bert-base-uncased', choices = ["bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased"])
    parser.add_argument("--seed", help = "seed", default = 100, type = int)
    parser.add_argument("--epoch", default = 5, type = int, help = "number epochs in training a model")
    parser.add_argument("--lr", default = 1e-5, type = float, help = "learning rate")
    parser.add_argument("--model_save_dir", default = "model_output", help = "Directory where the trained model would be saved")
    parser.add_argument("--batch_size", type = int, default = 50, help = "Training batch size")
    parser.add_argument("--to_shuffle", type = int, default = 1, choices = [1,0], help = "whether to shuffle the training data")
    parser.add_argument("--max_grad_norm", default = 1.00, type = float, help = "max gradient norm to clip")
    
    # ------- arguments for predicting over unseen data using a trained model ----
    parser.add_argument("--predict_str", default = None, help = " A string you want to make predictions for")
    parser.add_argument("--predict_infile", default = None, help = "A file containing lines of strings you want to make predictions for")
    parser.add_argument("--predict_outfile", default = None, help = "A file that saves the predicitons")

    args = parser.parse_args()
    
    # verify arguments
    if args.train: 
        assert args.unlabeled_event2sentis
        assert args.senti2polar
        assert args.train_dir
    
    if args.test:
        assert args.test_dir 
        assert args.checkpoint_dir

    if args.predict:
        assert args.checkpoint_dir
        assert args.predict_str is not None or args.predict_infile
    
    args.num_classes = len(label2ind)

    return args
    
def do_predict(checkpoint_dir, predict_str, predict_infile):
    train_vars = load_train_vars(checkpoint_dir)
    model_type = train_vars["model_type"]
    model = Model(model_type)
    model.load_state_dict(torch.load(join(checkpoint_dir, "model.ckpt")))

    if predict_str is not None: 
        strs = [predict_str] 
    else:
        with open(predict_infile) as f:
            strs = [line.strip() for line in f if line.strip() != ""]

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    input_ids = [tokenizer.encode(s) for s in strs]
    data = {
    "events":strs,
    "input_ids": input_ids
    }
    uloader = DataLoader(Event_Dataset(data, shuffle = False),batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
    
    pred_events, pred_probs, pred_Y, _ = predict(model, uloader)
    
    # make scores into the 0-100 scale
    pred_probs = [[k*100 for k in prob] for prob in pred_probs]
    if predict_str:
        # print out the one result
        print(f"""
            INPUT: {pred_events[0]}
            OUTPUT: {ind2label[pred_Y[0]]}
            SCORE: pos {pred_probs[0][label2ind['pos']]:.2f}, neg {pred_probs[0][label2ind['neg']]:.2f}, neu {pred_probs[0][label2ind['neu']]:.2f} 
            """)

    out_dict = [
        (
            e, 
            {   'pos': pred_prob[label2ind['pos']], 
                'neg': pred_prob[label2ind['neg']],
                'neu': pred_prob[label2ind['neu']]
            }, 
            ind2label[pred_y]
        ) for e, pred_prob, pred_y in zip(pred_events, pred_probs, pred_Y)
        ]


    if args.predict_infile:
        predict_outfile = args.predict_infile + ".json" if args.predict_outfile is None else args.predict_outfile
    else:
        predict_outfile = "out.json"

    with open(predict_outfile, 'w') as f:
        json.dump(out_dict, f, indent = 2)

    print(f"Predictions are saved to {predict_outfile}")




if __name__ == "__main__":

    args = get_args()
    print(f"{args}\n")

    seed = args.seed
    seed_everything(seed)

    if args.train: 
        do_training(args)

    if args.test:
        print("Testing")
        test_model(args.checkpoint_dir, join(args.test_dir, "labeled_events.json"))
    if args.predict:
        print("Predicting")
        do_predict(args.checkpoint_dir, args.predict_str, args.predict_infile)


