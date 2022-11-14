python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_1.pkl --discretization 1; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_1.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_1 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_2.pkl --discretization 2; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_2.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_2 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_3.pkl --discretization 3; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_3.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_3 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_5.pkl --discretization 5; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_5.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_5 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_10.pkl --discretization 10; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_10.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_10 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_25.pkl --discretization 25; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_25.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_25 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_50.pkl --discretization 50; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_50.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_50 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_100.pkl --discretization 100; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_100.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_100 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_250.pkl --discretization 250; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_250.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_250 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_500.pkl --discretization 500; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_500.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_500 uniform 2.0 --device cuda; \
python train.py breast_cancer uniform 2.0 --output_path ./models/ablation_discretizations_tabular/breast_cancer_l1 --model_name stump_ensemble_1000.pkl --discretization 1000; \
python certify.py breast_cancer ./models/ablation_discretizations_tabular/breast_cancer_l1/stump_ensemble_1000.pkl ./experiments/ablation_discretizations_tabular/breast_cancer_l1 certified_radii_1000 uniform 2.0 --device cuda; \
python analyze.py ./experiments/ablation_discretizations_tabular/breast_cancer_l1/table 4 ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_1  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_2  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_3  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_5  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_10  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_25  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_50  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_100  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_250  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_500  ./experiments/ablation_discretizations_tabular/breast_cancer_l1/certified_radii_1000;