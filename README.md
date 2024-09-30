## Instructions

1. Launch the batch inference [template](https://console.anyscale.com/v2/template-preview/batch-llm)
<img width="536" alt="image1" src="https://github.com/user-attachments/assets/c47183bb-2865-4adb-87ff-d46be3296f46">


2. Update the Head Node Type to the desired machine type
<img width="357" alt="image2" src="https://github.com/user-attachments/assets/cbb37e50-4040-4088-94f3-792f0f02551d">


| Model | Node Type |
| :---- | :---- |
| neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 | g6e.12xlarge |
| neuralmagic/Meta-Llama-3.1-7B-Instruct-FP8 | g6e.xlarge |


3. Run the following script on the template:

```
bash run_70b.sh 
# bash run_8b.sh
```

