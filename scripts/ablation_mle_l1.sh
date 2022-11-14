python train.py mnist_2_6 default 1.00 --output_path ./models/ablation_mle/mnist_2_6_l1_1.00 --model_name stump_ensemble_default.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_1.00/stump_ensemble_default.pkl ./experiments/ablation_mle/mnist_2_6_l1_1.00 certified_radii_default uniform 1.00 --device cuda; \
python train.py mnist_2_6 default 1.00 --output_path ./models/ablation_mle/mnist_2_6_l1_1.00 --model_name stump_ensemble_sampling.pkl --use_noisy_samples 2; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_1.00/stump_ensemble_sampling.pkl ./experiments/ablation_mle/mnist_2_6_l1_1.00 certified_radii_sampling uniform 1.00 --device cuda; \
python train.py mnist_2_6 uniform 1.00 --output_path ./models/ablation_mle/mnist_2_6_l1_1.00 --model_name stump_ensemble_mle.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_1.00/stump_ensemble_mle.pkl ./experiments/ablation_mle/mnist_2_6_l1_1.00 certified_radii_mle uniform 1.00 --device cuda; \
python analyze.py ./experiments/ablation_mle/mnist_2_6_l1_1.00/table 2 ./experiments/ablation_mle/mnist_2_6_l1_1.00/certified_radii_default ./experiments/ablation_mle/mnist_2_6_l1_1.00/certified_radii_sampling ./experiments/ablation_mle/mnist_2_6_l1_1.00/certified_radii_mle; \
python train.py mnist_2_6 default 4.00 --output_path ./models/ablation_mle/mnist_2_6_l1_4.00 --model_name stump_ensemble_default.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_4.00/stump_ensemble_default.pkl ./experiments/ablation_mle/mnist_2_6_l1_4.00 certified_radii_default uniform 4.00 --device cuda; \
python train.py mnist_2_6 default 4.00 --output_path ./models/ablation_mle/mnist_2_6_l1_4.00 --model_name stump_ensemble_sampling.pkl --use_noisy_samples 2; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_4.00/stump_ensemble_sampling.pkl ./experiments/ablation_mle/mnist_2_6_l1_4.00 certified_radii_sampling uniform 4.00 --device cuda; \
python train.py mnist_2_6 uniform 4.00 --output_path ./models/ablation_mle/mnist_2_6_l1_4.00 --model_name stump_ensemble_mle.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_4.00/stump_ensemble_mle.pkl ./experiments/ablation_mle/mnist_2_6_l1_4.00 certified_radii_mle uniform 4.00 --device cuda; \
python analyze.py ./experiments/ablation_mle/mnist_2_6_l1_4.00/table 2 ./experiments/ablation_mle/mnist_2_6_l1_4.00/certified_radii_default ./experiments/ablation_mle/mnist_2_6_l1_4.00/certified_radii_sampling ./experiments/ablation_mle/mnist_2_6_l1_4.00/certified_radii_mle; \
python train.py mnist_2_6 default 16.00 --output_path ./models/ablation_mle/mnist_2_6_l1_16.00 --model_name stump_ensemble_default.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_16.00/stump_ensemble_default.pkl ./experiments/ablation_mle/mnist_2_6_l1_16.00 certified_radii_default uniform 16.00 --device cuda; \
python train.py mnist_2_6 default 16.00 --output_path ./models/ablation_mle/mnist_2_6_l1_16.00 --model_name stump_ensemble_sampling.pkl --use_noisy_samples 2; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_16.00/stump_ensemble_sampling.pkl ./experiments/ablation_mle/mnist_2_6_l1_16.00 certified_radii_sampling uniform 16.00 --device cuda; \
python train.py mnist_2_6 uniform 16.00 --output_path ./models/ablation_mle/mnist_2_6_l1_16.00 --model_name stump_ensemble_mle.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l1_16.00/stump_ensemble_mle.pkl ./experiments/ablation_mle/mnist_2_6_l1_16.00 certified_radii_mle uniform 16.00 --device cuda; \
python analyze.py ./experiments/ablation_mle/mnist_2_6_l1_16.00/table 2 ./experiments/ablation_mle/mnist_2_6_l1_16.00/certified_radii_default ./experiments/ablation_mle/mnist_2_6_l1_16.00/certified_radii_sampling ./experiments/ablation_mle/mnist_2_6_l1_16.00/certified_radii_mle;
