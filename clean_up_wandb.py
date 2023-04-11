import wandb


api = wandb.Api()




artifact_path = "matezzzz/image_outpainting_maskgit/run_xk236zct_model"

keep_each = 20
delete_from = 0
delete_to = 57


for i in range(delete_from, delete_to+1):
    if i % keep_each == 0:
        print ("Keeping", i)
        continue
    try:
        a = api.artifact(f'{artifact_path}:v{i}')
        print ("Deleting", i)
    except wandb.errors.CommError:
        print ("Can't find", i)
        continue
    a.delete(delete_aliases=True)
    a.delete()