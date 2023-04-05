import wandb


api = wandb.Api()


for i in range(195):
    #print(i)
    try:
        a = api.artifact(f'matezzzz/image_outpainting_tokenizer/run_2llyronh_model:v{i}')
        print ("Deleting", i)
    except wandb.errors.CommError:
        print ("Can't find", i)
        continue    
    a.delete(delete_aliases=True)
    a.delete()