# METCS777-GenAIForTheme
This repository is a project for METCS777. This project focuses on fine-tuning Gen AI models for theme specific content

## Developer Install
1. Download `aws-cli`
2. create virtual env
3. run
   ```bash
   pip install -r requirements.txt
   ```

## Git Commit and SageMaker instruction

To facilitate colaboration, we can push our notebooks and codes to GitHub. Then Sagemaker server (Jupyter Lab) in each individual accounts can pull from our central Github Repository. To setup:
1. In your GitHub setting, create a personal access token, follow instruction [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
2. In AWS console, navigate to SageMaker service -> Notebook -> Git Repository. Follow this intruction from AWS to connect to GitRepository
![alt text](/images/sagemaker.png)
3. Now, when we launch Jupyter Lab and Jupyter Notebook, we can pull/push from repository using the Terminal, or by executing
```
!git pull origin main
```
in the notebook cell

https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-notebooks-now-support-git-integration-for-increased-persistence-collaboration-and-reproducibility/

### If you are using the lab
1. In your labmodule, get your credentials detail and follow instruction to add to `~/.aws/credentials` config
![alt text](./images/image.png)