# Learning to Censor by Noisy Sampling


This is the code for ECCV Submission - Paper ID: 3755.


## Installation
We strongly recommend working with <a href="https://hub.docker.com/search/?type=edition&offering=community" target="_blank">Docker Engine</a> and <a href="https://github.com/NVIDIA/nvidia-docker/tree/master">Nvidia-Docker</a>.
At this moment, the container can only run on a CUDA (_linux/amd64_) enabled machine due to specific compiled ops from <a href="https://github.com/erikwijmans/Pointnet2_PyTorch">Pointnet2_PyTorch</a>.

### Pull and run the Docker container
```bash
docker pull asafmanor/pytorch:samplenetreg_torch1.4
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name samplenetreg asafmanor/pytorch:samplenetreg_torch1.4
```

### Alternatively, build your own Docker image
#### On the host machine
```bash
docker build -t samplenetreg_torch1.4_image .
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name samplenetreg samplenetreg_torch1.4_image
```
#### Inside the Docker container
```bash
cd /root/Pointnet2_PyTorch
git checkout 5ff4382f56a8cbed2b5edd3572f97436271aba89
pip install -r requirements.txt
pip install -e .
cd /workspace
```

## Usage
### Data preparation

#### FaceScape
Download the facescape data from this <a href="https://facescape.nju.edu.cn/">website</a> to a desired folder to read from. Run `data/Preprocess_Data_v2.ipynb` to sample pointclouds (`.npy`) from mesh (`.obj`) files and generate annotations for the same. 
Kindly change the read/write paths in the notebook for proper execution.


### Training and evaluating
#### Facescape
For a quick start please use the scripts provided in `scripts/` directory. Run all scripts from the root directory of this repository. Note that a temporary `--base-name` is added to all scripts to maintain anonymity and needs be modified before running the scripts (parent directory containing annotation file and npy data after preprocessing).

1. Train base pointnet classifiers + finetune samplers:
   ` bash ./facescape/scripts/baseline/gender.sh `
   ` bash ./facescape/scripts/baseline/exp.sh `

2. Train with proxy attacker model for privacy + finetune worst case attacker: 
	a. <b>CBSN (Our Method)</b>
   		`bash ./facescape/scripts/cbns/exp_gender_best.sh` 
	b. Line Cloud
		`bash ./facescape/scripts/line/exp_gender_best.sh` 
	c. AS-AN
		`bash ./facescape/scripts/as_an/exp_gender_best.sh` 
	d. AS-ON
		`bash ./facescape/scripts/as_on/exp_gender_best.sh` 
	e. OS-AN
		`bash ./facescape/scripts/os_an/exp_gender_best.sh`

Results for the experiment will be collected in the provided log-directory (eg: `./facescape/log/cbns/finetune/test.log`). In order to reproduce the normalized hypervolume, a sweep over the hyperparameters to regulate privacy-utility is required which may take a long time to compute. For those interested, the parameters can easily be regulated by modifying the arguments the bash scripts (`./facescape/scripts/`). 

The resulting point clouds can be visualized using `results/plot_pointclouds.ipynb` with appropriate paths to trained model files. 

## Acknowledgment
This code builds upon the code provided in <a href="https://github.com/itailang/SampleNet">Samplenet</a>, <a href="https://github.com/hmgoforth/PointNetLK">PointNetLK</a>, <a href="https://github.com/erikwijmans/Pointnet2_PyTorch">Pointnet2_PyTorch</a> and <a href="https://github.com/unlimblue/KNN_CUDA">KNN_CUDA</a>. We thank the authors for sharing their code.

