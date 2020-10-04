import yaml
import argparse
import os


# read config.yaml file
def readYaml(path, args):
    if not os.path.exists(path):
        return args
    f = open(path)
    config = yaml.load(f)

    args.dataset = str(config["dataset"])
    args.clients = int(config["clients"])
    args.client_port = int(config["client_port"])
    args.data_size = config["data_size"]
    args.split_mode = str(config["split_mode"])
    args.label_num = config["label_num"]
    args.add_error = config["add_error"]
    args.error_rate = float(config["error_rate"])
    args.server_data = bool(config["server_data"])
    args.iid_share = bool(config["iid_share"])
    args.iid_rate = float(config["iid_rate"])
    args.type = str(config["type"])
    args.share_samples = int(config["share_samples"])
    args.global_dataset = bool(config["global_dataset"])
    args.data_rate = float(config["data_rate"])
    args.batch_size = int(config["batch_size"])
    args.test_batch_size = int(config["test_batch_size"])
    args.training_rounds = int(config["training_rounds"])
    args.federate_after_n_batches = int(config["federate_after_n_batches"])
    args.lr = float(config["lr"])
    args.cuda = bool(config["cuda"])
    args.seed = int(config["seed"])
    args.save_model = bool(config["save_model"])
    args.verbose = bool(config["verbose"])
    args.eval_every = int(config["eval_every"])
    args.fraction_client = float(config["fraction_client"])
    args.model = str(config["model"])
    args.aggregation = str(config["aggregation"])
    args.loss = str(config["loss"])
    args.optimizer = str(config["optimizer"])
    args.max_nr_batches = int(config["max_nr_batches"])




    return args
# function of arguments
def defineArgs():
    parser = argparse.ArgumentParser(description="parameters.")
     #dataset
    parser.add_argument('--dataset', '-d', type=str, default="mnist", help="dataset mnist or cifar10 (default=mnist)")
     #number of clients
    parser.add_argument('--clients', '-c', type=int, default=5,help="number of clients (default=5)")
     #client port
    parser.add_argument('--client_port', '-cp', type=int, default=8000,help="the client port (default=8000)")
     #the server hold data or not 
    parser.add_argument('--server_data', '-sd', type=bool, default=False,help="the server hold also a part of data or not (default=False)")
     #split mode iid or non iid
    parser.add_argument('--split_mode', '-sm', type=str, default="iid",help="split mode: 'iid' or 'niid' (default=iid)")
     #in the case of iid split mode we can choose if clients have a shared sampels and with what percentage        
    parser.add_argument('--iid_share', '-is', type=bool, default=False,help="--iid_share=true to share samples between clients in the iid split mode (default=False)")
     #the percentage of samples to share
    parser.add_argument('--iid_rate', '-ir', type=float, default=0.1, help="the percentage of samples to share between clients in the iid split mode (default=0.1)")
     #type of split in the case of the non iid split
    parser.add_argument('--type', '-t', type=str, default="random",help="in the non iid split there are 2 type : 'random' and 'label' for split by label (default=random)")
     #the number or the percentage of samples for each client in the case of the random split
    parser.add_argument('--data_size', '-ds', nargs='+',type=int, default=[1000,1000,1000,1000,1000],
                        help="the number of samples for each client ( default=[1000,1000,1000,1000,1000])")
     #number of label for each client
    parser.add_argument('--label_num', '-ln', nargs='+',type=int, default=[1,4,2,4,3], help="the number of labels for each client (default=[1,4,2,4,3] ")
     #the way of sharing samples of the same class
    parser.add_argument('--share_samples','-ss', type=int, default=0,help="how to share samples between clients who hold the same classes:\n 0: Share the same samples of class \n \t. 1: Share samples of class randomly \n .2: Share different samples of class ")
     #add a percentage for the global dataset to all clients  
    parser.add_argument('--global_dataset','-gd', type=bool, default=False,help="add a percentage for the global dataset for all clients (default=False)")
     #add some error to our data  
    parser.add_argument('--add_error','-ae', type=bool, default=False,help="add some error on data for each client withe a rate")
     #the percentage of data which we want to add
    parser.add_argument('--data_rate', '-dr', type=float, default=0.1,help="the percentage of data which we want to add (default 0.1)")
     #the percentage of error which we want to add
    parser.add_argument('--error_rate', '-er', type=float, default=0.01,help="the percentage of error which we want to add (default 0.01)")
     #import args form config.yaml file
    parser.add_argument('--config_file','-f',type=str,help="the path of your config file")


    args = parser.parse_args()

    if args.config_file != None:
        args = readYaml(args.config_file, args)

    # valid parameters
    dataset = ["cifar10","mnist","fashionmnist","sent140","shakespeare"]
    if args.dataset not in dataset:
        print("currently we support only : ",dataset)
        return

    if args.split_mode != "iid" and args.split_mode != "niid":
        print("there is two mode of split iid and niid ")
        return
    if args.type != "label" and args.type != "random":
        print("in the niid split mode there are two type of split label and random")
        return


    if args.split_mode == "niid":
        if args.type == "label":
            if args.server_data == True and args.clients != len(args.label_num)-1:
                print("Error: so such server_data is True he length of label_num has to equal tothe clients number +1, and the last case is for the server")
                  
            elif  args.server_data == False and args.clients != len(args.label_num):
                print("Error: the number of label_num has to equal to the clients number")
                return
            
        elif args.type == "random":
            if args.server_data == True:
                if len(args.data_size)!= args.clients+1:
                    print("Error: so such server_data is True he length of dataset_size has to equal to the clients number +1, and the last case is for the server")
                    return
            else :
                if len(args.data_size) != args.clients:
                    print("Error: the number of data_size has to equal to the clients number")
                    return

            return args
        return args
    return args

def trainArgs():
    parser = argparse.ArgumentParser(description="run federated learning using websocket client workers.")
    parser.add_argument('--clients', '-c', type=int, default=5,help="number of clients (default=5)")
    parser.add_argument("--batch_size","-bs", type=int, default=32, help="batch size of the training")
    parser.add_argument("--max_nr_batches","-mbs", type=int, default=10, help="max nbr of batch of the training")
    parser.add_argument("--test_batch_size","-ts", type=int, default=128, help="batch size used for the test data")
    parser.add_argument("--training_rounds","-tr", type=int, default=2, help="number of federated learning rounds")
    parser.add_argument("--federate_after_n_batches","-fb",type=int,default=10,help="number of training steps performed on each remote worker before averaging",)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model","-smdl", action="store_true", help="if set, model will be saved")
    parser.add_argument("--verbose","-v",action="store_true",help="if set, websocket client workers will be started in verbose mode",)
    parser.add_argument("--dataset","-d", default="mnist", help="dataset holded by users")
    parser.add_argument("--eval_every","-ee", type=int, default=10, help="evaluate the model evrey n rounds")
    parser.add_argument("--fraction_client","-fc", type=float, default=1, help="Number of clients that will in each round")
    parser.add_argument("--model","-m", default="cnn_mnist", help="the model that we will use.")
    parser.add_argument("--loss","-l", default="nll_loss", help="the loss function  nll_loss or cross_entropy") 
    parser.add_argument("--optimizer","-o", default="SGD", help="the optimizer that we will use : SGD or Adam")   
    parser.add_argument("--aggregation","-a", type=str, default="federated_avg", help="type of aggregation : federated_avg or wieghted_avg")
    parser.add_argument('--data_size', '-ds', nargs='+',type=int, default=[1000,1000,1000,1000,1000],
                        help="the number of samples for each client ( default=[1000,1000,1000,1000,1000]) ")        
    parser.add_argument('--config_file','-f',type=str,help="the path of the config file")
    parser.add_argument('--client_port', '-cp', type=int, default=8000,help="the client port (default=8000)")

     #split mode iid or non iid
    parser.add_argument('--split_mode', '-sm', type=str, default="iid",help="split mode: 'iid' or 'niid' (default=iid)")
    args = parser.parse_args()
    if args.config_file != None:
        args = readYaml(args.config_file, args)

    return args