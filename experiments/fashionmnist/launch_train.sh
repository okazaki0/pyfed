cd ..
cd ..
while true; do
	printf " 0)iid \n 1)random non_iid \n 2)label non_iid (type 0) \n 3)label non_iid (type 1) \n 4)label non_iid (type 2) \n 5)exit\n"
    read -p "Wish config do you want to use ? " yn
    case $yn in
        [0]* ) python3 run/training/main.py --config_path experiments/fashionmnist/config/config_iid.yaml ; break;;
        [1]* ) python3 run/training/main.py --config_path experiments/fashionmnist/config/config_niid_random.yaml ; break;;
        [2]* ) python3 run/training/main.py --config_path experiments/fashionmnist/config/config_niid_label_t0.yaml ; break;;
        [3]* ) python3 run/training/main.py --config_path experiments/fashionmnist/config/config_niid_label_t1.yaml ; break;;
        [4]* ) python3 run/training/main.py --config_path experiments/fashionmnist/config/config_niid_label_t2.yaml ; break;;
        [5]* ) exit;;
        * ) echo "Please choose number.";;
    esac
done