# This scripts runs all major algorithms with toy datasets
# Must be launched and run from archai root directory

# Darts toy end-to-end script
echo "----------DARTS Toy End-2-end------------"
echo "-----------------------------------------"
python scripts/darts/cifar_e2e.py \
--config confs/cifar_toy.yaml \
--config-defaults confs/darts_cifar.yaml

# Petridish toy end-to-end script
echo "----------Petridish Toy End-2-end------------"
echo "-----------------------------------------"
python scripts/petridish/cifar_e2e.py \
--config confs/cifar_toy.yaml \
--config-defaults confs/petridish_cifar.yaml

# Random toy end-to-end script
echo "----------Random Toy End-2-end------------"
echo "-----------------------------------------"
python scripts/random/cifar_e2e.py \
--config confs/cifar_toy.yaml \
--config-defaults confs/random_cifar.yaml



