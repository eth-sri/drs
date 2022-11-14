python train.py breast_cancer uniform 2.00 --output_path ./models/independent/breast_cancer_l1 --model_name stump_ensemble.pkl; \
python certify.py breast_cancer ./models/independent/breast_cancer_l1/stump_ensemble.pkl ./experiments/independent/breast_cancer_l1 certified_radii uniform 2.00 --device cuda; \
python analyze.py ./experiments/independent/breast_cancer_l1/table 0 ./experiments/independent/breast_cancer_l1/certified_radii; \
python train.py diabetes uniform 0.35 --output_path ./models/independent/diabetes_l1 --model_name stump_ensemble.pkl; \
python certify.py diabetes ./models/independent/diabetes_l1/stump_ensemble.pkl ./experiments/independent/diabetes_l1 certified_radii uniform 0.35 --device cuda; \
python analyze.py ./experiments/independent/diabetes_l1/table 0 ./experiments/independent/diabetes_l1/certified_radii; \
python train.py fmnist_sandal_sneaker uniform 4.00 --output_path ./models/independent/fmnist_sandal_sneaker_l1 --model_name stump_ensemble.pkl; \
python certify.py fmnist_sandal_sneaker ./models/independent/fmnist_sandal_sneaker_l1/stump_ensemble.pkl ./experiments/independent/fmnist_sandal_sneaker_l1 certified_radii uniform 4.00 --device cuda; \
python analyze.py ./experiments/independent/fmnist_sandal_sneaker_l1/table 0 ./experiments/independent/fmnist_sandal_sneaker_l1/certified_radii; \
python train.py mnist_1_5 uniform 4.00 --output_path ./models/independent/mnist_1_5_l1 --model_name stump_ensemble.pkl; \
python certify.py mnist_1_5 ./models/independent/mnist_1_5_l1/stump_ensemble.pkl ./experiments/independent/mnist_1_5_l1 certified_radii uniform 4.00 --device cuda; \
python analyze.py ./experiments/independent/mnist_1_5_l1/table 0 ./experiments/independent/mnist_1_5_l1/certified_radii; \
python train.py mnist_2_6 uniform 4.00 --output_path ./models/independent/mnist_2_6_l1 --model_name stump_ensemble.pkl; \
python certify.py mnist_2_6 ./models/independent/mnist_2_6_l1/stump_ensemble.pkl ./experiments/independent/mnist_2_6_l1 certified_radii uniform 4.00 --device cuda; \
python analyze.py ./experiments/independent/mnist_2_6_l1/table 0 ./experiments/independent/mnist_2_6_l1/certified_radii;
